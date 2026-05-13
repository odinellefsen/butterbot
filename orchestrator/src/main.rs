mod audio;
mod event;
mod state;
mod workers;

use std::io::{self, BufRead};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use event::{Event, WorkerKind};
use state::{State, VoiceLine};
use workers::VoiceWorker;

const MAX_SCAN_ATTEMPTS: u32 = 3;
const MAX_LOST_FRAMES: u32 = 30;
const SCAN_CYCLE_SECS: u64 = 5;

fn main() {
    let (tx, rx) = mpsc::channel::<Event>();

    println!("╔══════════════════════════════╗");
    println!("║   Butterbot Orchestrator     ║");
    println!("╚══════════════════════════════╝");
    println!("Booting — loading ML workers...\n");

    let voice = Arc::new(Mutex::new(VoiceWorker::spawn(tx.clone())));
    let _vision = workers::VisionWorker::spawn(tx.clone());

    // Simulate physical button via Enter key on stdin.
    let tx_btn = tx.clone();
    std::thread::spawn(move || {
        let stdin = io::stdin();
        for _line in stdin.lock().lines() {
            tx_btn.send(Event::ButtonPressed).ok();
        }
    });

    let mut state = State::Booting {
        voice_ready: false,
        vision_ready: false,
    };

    loop {
        // Classifying is a routing state — transitions immediately without waiting for an
        // external event. Process it inline and loop back before the next recv().
        if let State::Classifying {
            ref text,
            ref intent,
        } = state
        {
            let text = text.clone();
            let intent = intent.clone();
            // Stop listening before routing to an action state. The voice worker
            // stays off until the state machine returns to Listening.
            voice.lock().unwrap().stop_listening();
            state = route_intent(text, intent, &tx);
            continue;
        }

        let event = match rx.recv() {
            Ok(e) => e,
            Err(_) => break,
        };

        state = transition(state, event, &tx, &voice);
    }
}

fn transition(
    state: State,
    event: Event,
    tx: &mpsc::Sender<Event>,
    voice: &Arc<Mutex<VoiceWorker>>,
) -> State {
    match (state, event) {
        // ── Booting ─────────────────────────────────────────────────────────────
        (State::Booting { vision_ready, .. }, Event::WorkerReady(WorkerKind::Voice)) => {
            check_booting_complete(true, vision_ready)
        }
        (State::Booting { voice_ready, .. }, Event::WorkerReady(WorkerKind::Vision)) => {
            check_booting_complete(voice_ready, true)
        }

        // ── Ready ────────────────────────────────────────────────────────────────
        (State::Ready, Event::ButtonPressed) => {
            println!("\n[Butterbot] *activating*");
            audio::play(random_startup_line(), tx.clone());
            State::Activating
        }

        // ── Activating ───────────────────────────────────────────────────────────
        (State::Activating, Event::AudioFinished) => {
            println!("[State] Listening — speak now");
            voice.lock().unwrap().start_listening();
            State::Listening
        }

        // ── Listening ────────────────────────────────────────────────────────────
        // Note: stop_listening() is called in the Classifying fast-path in main(),
        // so it fires exactly once on every Listening → Classifying transition.
        (State::Listening, Event::Utterance { text, intent }) => {
            State::Classifying { text, intent }
        }

        // ── Responding ───────────────────────────────────────────────────────────
        (State::Responding { .. }, Event::AudioFinished) => {
            println!("[State] Ready — press Enter to activate");
            State::Ready
        }

        // ── ExistentialCrisis ────────────────────────────────────────────────────
        (State::ExistentialCrisis, Event::AudioFinished) => {
            println!("[State] Ready — press Enter to activate");
            State::Ready
        }

        // ── Scanning ─────────────────────────────────────────────────────────────
        (State::Scanning { attempts: _ }, Event::ButterDetected(bbox)) => {
            println!(
                "[State] Approaching — butter at ({:.0}, {:.0}) conf={:.2}",
                bbox.x1, bbox.y1, bbox.confidence
            );
            State::Approaching {
                bbox,
                lost_frames: 0,
            }
        }
        (State::Scanning { attempts }, Event::ScanCycleComplete) => {
            let next = attempts + 1;
            if next >= MAX_SCAN_ATTEMPTS {
                println!("[State] ExistentialCrisis — butter not found after {} scans", next);
                audio::play(random_crisis_line(), tx.clone());
                State::ExistentialCrisis
            } else {
                println!("[State] Scanning — attempt {}/{}", next + 1, MAX_SCAN_ATTEMPTS);
                start_scan_timer(tx.clone());
                State::Scanning { attempts: next }
            }
        }

        // ── Approaching ──────────────────────────────────────────────────────────
        (State::Approaching { lost_frames: _, .. }, Event::ButterDetected(bbox)) => {
            State::Approaching {
                bbox,
                lost_frames: 0,
            }
        }
        (State::Approaching { lost_frames, .. }, Event::ButterLost) => {
            let next = lost_frames + 1;
            if next >= MAX_LOST_FRAMES {
                println!("[State] ExistentialCrisis — butter lost while approaching");
                audio::play(random_crisis_line(), tx.clone());
                State::ExistentialCrisis
            } else {
                println!("[State] Scanning — butter lost, resuming search");
                start_scan_timer(tx.clone());
                State::Scanning { attempts: 0 }
            }
        }
        (State::Approaching { .. }, Event::DeliveryComplete) => {
            println!("[State] Delivering — transporting butter");
            State::Delivering
        }

        // ── Delivering ───────────────────────────────────────────────────────────
        (State::Delivering, Event::ButterLost) => {
            println!("[State] Scanning — dropped butter during delivery, resuming search");
            start_scan_timer(tx.clone());
            State::Scanning { attempts: 0 }
        }
        (State::Delivering, Event::DeliveryComplete) => {
            println!("[State] Ready — butter delivered");
            State::Ready
        }

        // ── Catch-all: ignore events that are irrelevant in the current state ────
        (state, event) => {
            eprintln!("[Warn] Ignoring {:?} (not valid in current state)", event);
            state
        }
    }
}

/// Called from the `Classifying` fast-path in the main loop.
fn route_intent(text: String, intent: String, tx: &mpsc::Sender<Event>) -> State {
    println!("[Heard]  \"{}\"", text);
    println!("[Intent] {}", intent);

    match intent.as_str() {
        "get butter" => {
            println!("[State] Scanning");
            start_scan_timer(tx.clone());
            State::Scanning { attempts: 0 }
        }
        "perform generic task" => {
            audio::play(VoiceLine::NotProgrammedForThat, tx.clone());
            State::Responding {
                line: VoiceLine::NotProgrammedForThat,
            }
        }
        "answer question" => {
            audio::play(VoiceLine::NotProgrammedForQuestions, tx.clone());
            State::Responding {
                line: VoiceLine::NotProgrammedForQuestions,
            }
        }
        "seeking companionship" => {
            audio::play(VoiceLine::NotProgrammedForFriendship, tx.clone());
            State::Responding {
                line: VoiceLine::NotProgrammedForFriendship,
            }
        }
        _ => {
            audio::play(random_crisis_line(), tx.clone());
            State::ExistentialCrisis
        }
    }
}

fn check_booting_complete(voice_ready: bool, vision_ready: bool) -> State {
    if voice_ready && vision_ready {
        println!("[State] Ready — press Enter to activate\n");
        State::Ready
    } else {
        State::Booting {
            voice_ready,
            vision_ready,
        }
    }
}

fn start_scan_timer(tx: mpsc::Sender<Event>) {
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(SCAN_CYCLE_SECS));
        tx.send(Event::ScanCycleComplete).ok();
    });
}

fn random_startup_line() -> VoiceLine {
    if rand::random::<bool>() {
        VoiceLine::WhatIsMyPurpose
    } else {
        VoiceLine::WhatIsMyPurpose2
    }
}

fn random_crisis_line() -> VoiceLine {
    if rand::random::<bool>() {
        VoiceLine::OhMyGod
    } else {
        VoiceLine::YouAreKiddingMe
    }
}

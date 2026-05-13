use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::Sender;

use crate::event::{Event, WorkerKind};
use crate::state::BoundingBox;

pub struct Worker {
    process: Child,
}

impl Worker {
    /// Spawns the voice worker (Vosk + RoBERTa). Emits `WorkerReady` and `Utterance` events.
    pub fn spawn_voice(tx: Sender<Event>) -> Self {
        let mut process = Command::new("python3")
            .arg("../voice-pipeline/worker.py")
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("failed to spawn voice worker — is Python 3 available?");

        let stdout = process.stdout.take().unwrap();
        std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines().flatten() {
                match serde_json::from_str::<serde_json::Value>(&line) {
                    Ok(msg) => handle_voice_message(msg, &tx),
                    Err(e) => eprintln!("[Voice Worker] Bad JSON: {} — {}", line, e),
                }
            }
            eprintln!("[Voice Worker] Process exited unexpectedly");
        });

        Worker { process }
    }

    /// Spawns the vision worker (YOLOv8 ONNX). Emits `WorkerReady`, `ButterDetected`, and
    /// `ButterLost` events. On Mac without a camera this runs as a stub.
    pub fn spawn_vision(tx: Sender<Event>) -> Self {
        let mut process = Command::new("python3")
            .arg("../butter_detection_yolov8/worker.py")
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("failed to spawn vision worker — is Python 3 available?");

        let stdout = process.stdout.take().unwrap();
        std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines().flatten() {
                match serde_json::from_str::<serde_json::Value>(&line) {
                    Ok(msg) => handle_vision_message(msg, &tx),
                    Err(e) => eprintln!("[Vision Worker] Bad JSON: {} — {}", line, e),
                }
            }
            eprintln!("[Vision Worker] Process exited unexpectedly");
        });

        Worker { process }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.process.kill().ok();
    }
}

fn handle_voice_message(msg: serde_json::Value, tx: &Sender<Event>) {
    match msg["type"].as_str() {
        Some("ready") => {
            println!("[Voice Worker] Ready");
            tx.send(Event::WorkerReady(WorkerKind::Voice)).ok();
        }
        Some("utterance") => {
            let text = msg["text"].as_str().unwrap_or("").to_string();
            let intent = msg["intent"]
                .as_str()
                .unwrap_or("existential crisis")
                .to_string();
            println!("[Voice Worker] \"{}\" → {}", text, intent);
            tx.send(Event::Utterance { text, intent }).ok();
        }
        other => {
            eprintln!("[Voice Worker] Unknown message type: {:?}", other);
        }
    }
}

fn handle_vision_message(msg: serde_json::Value, tx: &Sender<Event>) {
    match msg["type"].as_str() {
        Some("ready") => {
            println!("[Vision Worker] Ready");
            tx.send(Event::WorkerReady(WorkerKind::Vision)).ok();
        }
        Some("butter_detected") => {
            let bbox = BoundingBox {
                x1: msg["bbox"]["x1"].as_f64().unwrap_or(0.0) as f32,
                y1: msg["bbox"]["y1"].as_f64().unwrap_or(0.0) as f32,
                x2: msg["bbox"]["x2"].as_f64().unwrap_or(0.0) as f32,
                y2: msg["bbox"]["y2"].as_f64().unwrap_or(0.0) as f32,
                confidence: msg["bbox"]["confidence"].as_f64().unwrap_or(0.0) as f32,
            };
            tx.send(Event::ButterDetected(bbox)).ok();
        }
        Some("butter_lost") => {
            tx.send(Event::ButterLost).ok();
        }
        Some("delivery_complete") => {
            tx.send(Event::DeliveryComplete).ok();
        }
        other => {
            eprintln!("[Vision Worker] Unknown message type: {:?}", other);
        }
    }
}

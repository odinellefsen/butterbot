use crate::state::BoundingBox;

#[derive(Debug)]
pub enum WorkerKind {
    Voice,
    Vision,
}

#[derive(Debug)]
pub enum Event {
    /// A Python worker has finished loading its models and is ready.
    WorkerReady(WorkerKind),

    /// Physical button press (or Enter key in dev mode).
    ButtonPressed,

    /// The currently playing audio clip has finished.
    AudioFinished,

    /// Vosk produced a complete utterance; RoBERTa has already classified it.
    Utterance { text: String, intent: String },

    /// Vision worker detected butter in the current frame.
    ButterDetected(BoundingBox),

    /// Vision worker reports no butter in the current frame.
    ButterLost,

    /// One full scan rotation has completed without finding butter.
    ScanCycleComplete,

    /// Robot has reached the user with the butter (distance sensor or timeout).
    DeliveryComplete,
}

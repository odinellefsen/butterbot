#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
}

#[allow(dead_code)]
impl BoundingBox {
    /// Normalised horizontal center (0.0 = left edge, 1.0 = right edge).
    pub fn center_x_norm(&self, frame_width: f32) -> f32 {
        ((self.x1 + self.x2) / 2.0) / frame_width
    }

    /// Fraction of frame area covered — used as a proxy for distance.
    pub fn area_fraction(&self, frame_w: f32, frame_h: f32) -> f32 {
        let w = (self.x2 - self.x1).abs();
        let h = (self.y2 - self.y1).abs();
        (w * h) / (frame_w * frame_h)
    }
}

#[derive(Debug, Clone)]
pub enum VoiceLine {
    WhatIsMyPurpose,
    WhatIsMyPurpose2,
    NotProgrammedForThat,
    NotProgrammedForQuestions,
    NotProgrammedForFriendship,
    OhMyGod,
    YouAreKiddingMe,
}

impl VoiceLine {
    pub fn filename(&self) -> &'static str {
        match self {
            VoiceLine::WhatIsMyPurpose => "what_is_my_purpose.mp3",
            VoiceLine::WhatIsMyPurpose2 => "what_is_my_purpose_2.mp3",
            VoiceLine::NotProgrammedForThat => "i_am_not_programmed_for_that.mp3",
            VoiceLine::NotProgrammedForQuestions => "i_am_not_programmed_for_questions.mp3",
            VoiceLine::NotProgrammedForFriendship => "i_am_not_programmed_for_friendship.mp3",
            VoiceLine::OhMyGod => "oh_my_god.mp3",
            VoiceLine::YouAreKiddingMe => "you_are_fucking_kidding_me.mp3",
        }
    }
}

#[derive(Debug)]
pub enum State {
    /// Workers are loading their ML models. Both flags must be true to proceed.
    Booting {
        voice_ready: bool,
        vision_ready: bool,
    },
    /// All workers ready. Waiting for the activation button press.
    Ready,
    /// Playing the "What is my purpose?" startup clip.
    Activating,
    /// Microphone open, streaming audio through Vosk, waiting for a full utterance.
    Listening,
    /// Utterance received; routing to the appropriate action state based on intent.
    Classifying {
        text: String,
        intent: String,
    },
    /// Playing a rejection or personality voice clip.
    Responding {
        #[allow(dead_code)]
        line: VoiceLine,
    },
    /// Playing an existential despair clip before resetting to Ready.
    ExistentialCrisis,
    /// Rotating and scanning the environment for butter.
    Scanning {
        attempts: u32,
    },
    /// Butter is in frame; driving toward it.
    Approaching {
        #[allow(dead_code)]
        bbox: BoundingBox,
        lost_frames: u32,
    },
    /// Butter acquired; transporting it back to the user.
    Delivering,
}

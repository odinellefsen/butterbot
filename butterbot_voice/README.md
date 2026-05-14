# butterbot_voice

Robot voice clips played by the orchestrator in response to classified intents. All clips
are MP3 files processed through Voicemod to give the robot a flat, mechanically distressed
synthetic voice.

## Clips

| File | Played when | State |
|---|---|---|
| `what_is_my_purpose.mp3` | Robot activates (variant 1) | `Activating` |
| `what_is_my_purpose_2.mp3` | Robot activates (variant 2) | `Activating` |
| `i_am_not_programmed_for_that.mp3` | Intent: `perform generic task` | `Responding` |
| `i_am_not_programmed_for_questions.mp3` | Intent: `answer question` | `Responding` |
| `i_am_not_programmed_for_friendship.mp3` | Intent: `seeking companionship` | `Responding` |
| `oh_my_god.mp3` | Existential crisis (variant 1) | `ExistentialCrisis` |
| `you_are_fucking_kidding_me.mp3` | Existential crisis (variant 2) | `ExistentialCrisis` |

The two startup variants and two crisis variants are chosen randomly on each playback.

## Adding new clips

1. Record or generate a new MP3 and place it in this directory.
2. Add a new variant to `VoiceLine` in `orchestrator/src/state.rs`:
   ```rust
   pub enum VoiceLine {
       // ...
       MyNewLine,
   }
   ```
3. Add the filename mapping in the `filename()` method:
   ```rust
   VoiceLine::MyNewLine => "my_new_line.mp3",
   ```
4. Use it in `orchestrator/src/main.rs` or `audio.rs` where appropriate.

## Playback

The orchestrator plays clips using `afplay` on macOS and `mpg123` on Linux/Raspberry Pi OS.

On Raspberry Pi OS, install `mpg123` if not already present:

```bash
sudo apt install mpg123
```

Playback is synchronous — `AudioFinished` is only emitted after the clip has fully played,
so the state machine never advances mid-clip.

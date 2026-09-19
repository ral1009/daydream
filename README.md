# Daydream
 
An interactive pogo platformer where you control the character with your own body: lean left and right to steer, and scream as loud as you can to jump.
A Python script watches your webcam and microphone, computes your lean angle and shout volume, and serves them to the Godot game over a local HTTP endpoint.
The Game was made in one day in Godot for Hack Club's Hackathon/GameJam Daydream, winning first place in votes and second place overall.
All art is custom made for this project in piskel.
 
**Tools:** Godot (GDScript) · Python (OpenCV, MediaPipe, Flask)
 
## How it works
 
- **`gigatest.py`** — reads your webcam feed (OpenCV + MediaPipe Pose) to compute your body's lean angle vs. vertical, and reads your microphone (via `sounddevice`) to track shout volume in dBFS. It shows a live camera skeleton overlay and a loudness meter (Pygame), and serves both values as JSON from a local Flask server at `http://127.0.0.1:8080/data`.
- **`node.gd`** — an autoloaded Godot script that polls that endpoint ~5 times a second and stores the latest angle/volume in `globals.gd`.
- **`player.gd`** — reads `Globals.angle_rad` to set the pogo stick's tilt (and therefore its bounce direction), and triggers a jump whenever `Globals.volume` crosses a threshold — i.e., whenever you yell loud enough.
## Requirements
 
- [Godot Engine](https://godotengine.org/download) 4.5+
- Python 3.9+
- A webcam and a microphone
- Python packages:
```bash
  pip install opencv-python mediapipe sounddevice pygame numpy flask
```
 
## Running it
 
1. Start the body/voice tracker first:
```bash
   python gigatest.py
```
   This opens a camera window (with skeleton overlay and lean angle) and a loudness meter window, and starts serving data on port 8080.
2. Open the project in Godot (`project.godot`) and press Play.
3. Lean your body left/right in front of the camera to steer, and yell to jump.
 
Useful keys in the `gigatest.py` windows: `↑`/`↓` adjust the yell-volume threshold, `R` resets the peak meter, `D` lists audio devices, `Q` quits.
 
## Controls
 
| Action | Input |
|---|---|
| Steer left / right | Lean your body left / right (webcam) |
| Jump / bounce higher | Yell above the volume threshold (mic) |
 
`Trash/player2.gd` and the (hidden) `Player2` node are leftovers from an earlier local 2-player, keyboard-controlled prototype and aren't used in the current single-player, body-controlled version.
 
## Project structure
 
```
gigatest.py         Webcam + mic tracker and Flask server
globals.gd          Autoloaded state (angle_rad, volume) shared across the game
node.gd             Polls gigatest.py's HTTP endpoint and updates globals.gd
player.gd           Player controller — tilt from angle, jump from volume
node_2d.tscn        Main scene — level layout, tilemap, player
project.godot        Project & input configuration
Sprites/
  Player/           Pogo stick sprites (grounded/crouch states)
  Tiles/            Grass, dirt, spike, cloud, and win tileset pieces
Trash/               Unused legacy assets from the earlier 2-player prototype
```
 
## Notes
 
This is an early prototype — expect rough edges, hardcoded values (e.g. volume threshold, mic strength), and placeholder art in `Trash/`.
 

extends CharacterBody2D

@export var micstrength: float = 50.0
@export var speed: float = 300.0
@export var gravity: float = 1200.0
@export var rotation_speed: float = 180.0
@export var grounded_display_time: float = 0.15

# Your volume scale uses dB (e.g., -40 .. -10). Jump when volume >= threshold.
@export var volume_threshold: float = -25.0
@export var jump_cooldown_ms: int = 150    # prevent rapid re-triggers while yelling

@onready var sprite: Sprite2D = $Sprite2D
@export var ground_texture: Texture2D
@export var air_texture: Texture2D

var grounded_time: float = 0.0
var prev_over: bool = false
var next_allowed_jump_ms: int = 0

func _ready() -> void:
	pass

func _physics_process(delta: float) -> void:
	print( Globals.volume)
	
	var on_floor := is_on_floor()

	# Gravity while airborne
	if not on_floor:
		velocity.y += gravity * delta

	# Drive tilt from HTTP angle (radians), clamped
	rotation = clampf(Globals.angle_rad, -0.61, 0.61)

	# Forward speed from tilt, always applied
	velocity.x = sin(rotation) * speed

	# ----- Volume-triggered jump (works midair) -----
	var over := Globals.volume >= volume_threshold
	var rising := over and not prev_over
	var now_ms := Time.get_ticks_msec()
	var did_jump := false

	# Also allow keyboard jump if you want: OR Input.is_action_just_pressed("ui_up")
	if rising and now_ms >= next_allowed_jump_ms:
		velocity.y = (-cos(rotation) * (400 + micstrength * (Globals.volume - volume_threshold)))
		grounded_time = grounded_display_time
		next_allowed_jump_ms = now_ms + jump_cooldown_ms
		did_jump = true

	prev_over = over

	# If on the floor and we didn't jump this frame, give a gentle push along the surface
	if on_floor and not did_jump:
		grounded_time = grounded_display_time
		velocity.y = -cos(rotation) * speed

	# Sprite swap with timer
	if grounded_time > 0.0:
		sprite.texture = ground_texture
		grounded_time -= delta
	else:
		sprite.texture = air_texture

	move_and_slide()

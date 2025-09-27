extends CharacterBody2D

@export var jump_strength: float = 400.0
@export var micstrength: float = 800.0
@export var speed: float = 400.0
@export var gravity: float = 1200.0
@export var rotation_speed: float = 180.0

@onready var sprite: Sprite2D = $Sprite2D
@export var ground_texture: Texture2D
@export var air_texture: Texture2D

var horizontal_velocity: float = 0.0
var grounded_time: float = 0.0
@export var grounded_display_time: float = 0.15   # seconds to keep ground sprite visible

func _physics_process(delta: float):
	# Midair: apply gravity
	if not is_on_floor():
		velocity.y += gravity * delta

		if Input.is_action_pressed("p2left"):
			if rotation > -0.61:
				rotation -= deg_to_rad(rotation_speed) * delta
		elif Input.is_action_pressed("p2right"):
			if rotation < 0.61:
				rotation += deg_to_rad(rotation_speed) * delta

	# On floor
	if is_on_floor():
		grounded_time = grounded_display_time   # refresh timer
		velocity.x = sin(rotation) * speed
		if Input.is_action_pressed("p2up"):
			velocity.y = -cos(rotation) * micstrength
		else:
			velocity.y = -cos(rotation) * speed

	# --- Sprite swap with timer ---
	if grounded_time > 0.0:
		sprite.texture = ground_texture
		grounded_time -= delta
	else:
		sprite.texture = air_texture

	move_and_slide()

extends CharacterBody2D

@export var jump_strength: float = 400.0       # Base jump speed
@export var micstrength: float = 800.0
@export var speed: float = 400.0               # Ground horizontal speed
@export var gravity: float = 1200.0
@export var rotation_speed: float = 180.0      # Degrees per second in midair

var horizontal_velocity: float = 0.0

func _physics_process(delta: float):
	# Midair: apply gravity
	if not is_on_floor():
		velocity.y += gravity * delta

		# Rotate in midair if holding left/right
		if Input.is_action_pressed("ui_left"):
			if rotation > -0.61:
				rotation -= deg_to_rad(rotation_speed) * delta
		elif Input.is_action_pressed("ui_right"):
			if rotation < 0.61:
				rotation += deg_to_rad(rotation_speed) * delta

		# Maintain horizontal momentum from last landing

	# On floor: apply bounce based on rotation
	if is_on_floor():
		# Optional: horizontal input on ground
		# Bounce based on rotation
		# Use sin for X component, cos for Y component
		velocity.x = sin(rotation) * speed
		if Input.is_action_pressed("ui_up"):
			velocity.y = -cos(rotation) * micstrength
		else:
			velocity.y = -cos(rotation) * speed  # negative Y is up
		print((rotation*180)/3.14)

	move_and_slide()

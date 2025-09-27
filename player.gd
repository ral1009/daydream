extends CharacterBody2D

@export var micstrength: float = 1100.0
@export var speed: float = 600.0
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

		if Input.is_action_pressed("ui_left"):
			if rotation > -0.61:
				rotation -= deg_to_rad(rotation_speed) * delta
		elif Input.is_action_pressed("ui_right"):
			if rotation < 0.61:
				rotation += deg_to_rad(rotation_speed) * delta

	# On floor
	if is_on_floor():
		grounded_time = grounded_display_time   # refresh timer
		velocity.x = sin(rotation) * speed
		if Input.is_action_pressed("ui_up"):
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
	for i in get_slide_collision_count():
		var collision = get_slide_collision(i)
		var collider = collision.get_collider()
		if collider and collider.is_in_group("spike"):
			die()


func die():
	var tree = Engine.get_main_loop() as SceneTree
	tree.change_scene_to_file("res://node_2d.tscn")

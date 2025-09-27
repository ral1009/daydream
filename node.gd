extends Node

signal http_data(angle_rad: float, volume: float)

var http_request: HTTPRequest

var angle_rad: float = 0.0
var volume: float = 0.0
var has_angle := false
var has_volume := false

func _ready() -> void:
	http_request = HTTPRequest.new()
	add_child(http_request)
	http_request.request_completed.connect(_on_request_completed)
	_poll_data()

func _poll_data() -> void:
	var err = http_request.request("http://127.0.0.1:8080/data")
	if err != OK:
		print("Error making request:", err)
	get_tree().create_timer(0.2).timeout.connect(_poll_data)  # poll ~5x/sec

func _on_request_completed(result, response_code, headers, body) -> void:
	if response_code != 200:
		print("HTTP error:", response_code)
		return

	var json := JSON.new()
	if json.parse(body.get_string_from_utf8()) != OK:
		print("JSON parse error")
		return

	var data = json.get_data()
	# Expecting: {"angle":"12.3°", "volume": 0.42} from your Flask server
	if "angle" in data:
		var s := str(data["angle"]).strip_edges()
		s = s.replace("°", "")
		if s != "":
			Globals.angle_rad = deg_to_rad(float(s))
			Globals.has_angle = true
			print(Globals.angle_rad)
			
	if "volume" in data:
		Globals.volume = float(data["volume"])
		Globals.has_volume = true
		print(Globals.volume)

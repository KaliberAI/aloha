import argparse
import socket
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from fastapi import FastAPI, HTTPException
import uvicorn

# State IDs (matches tracking_clean STATE_ID_MAP)
#   0 - SLEEP
#   1 - OPENING
#   2 - WAVING
#   3 - NOD
#   4 - TRACK


# 1. Define the ROS 2 Publisher Node
class RobotStatePublisher(Node):
    def __init__(self):
        super().__init__('web_state_bridge')
        # This matches the subscriber in your robot script
        self.publisher_ = self.create_publisher(Int32, 'robot_state', 10)

    def publish_state(self, state_id: int):
        msg = Int32()
        msg.data = state_id
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published state transition: {state_id}')

# 2. Setup FastAPI
app = FastAPI(title="Robot Command Center")
ros_node = None
current_state = 1  # Initial state is 1

@app.post("/set-state/{state_id}")
async def set_robot_state(state_id: int):
    global current_state
    
    # Validate against tracking_clean STATE_ID_MAP (0-4)
    if not (0 <= state_id <= 4):
        return {"status": "error", "detail": "Invalid State ID. Use 0-4."}
    
    # Guard transition from state 0 to any non-1 state
    if current_state == 0 and state_id != 1:
        return {"status": "error", "detail": "wake the robot first"}
    
    ros_node.publish_state(state_id)
    current_state = state_id
    return {"status": "success", "detail": f"State transitioned to {state_id}"}

def is_port_available(port):
    """Check if a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('0.0.0.0', port))
            return True
        except OSError:
            return False

def find_available_port(start_port, max_attempts=10):
    """Find an available port starting from start_port."""
    for i in range(max_attempts):
        port = start_port + i
        if is_port_available(port):
            return port
    return None

# 3. Main Execution with Threading
def main():
    parser = argparse.ArgumentParser(description='Robot Command Center Web Bridge')
    parser.add_argument('--port', '-p', type=int, default=8010,
                        help='Port to run the server on (default: 8010). If port is in use, will automatically find an available port.')
    args = parser.parse_args()
    
    # Check if port is available, find alternative if needed
    port = args.port
    if not is_port_available(port):
        print(f"Port {port} is already in use, searching for an available port...")
        available_port = find_available_port(port)
        if available_port:
            port = available_port
            print(f"Found available port: {port}")
        else:
            print(f"Error: Could not find an available port starting from {port}")
            return
    
    global ros_node
    rclpy.init()
    ros_node = RobotStatePublisher()

    # Run ROS 2 spin in a separate thread so it doesn't block FastAPI
    ros_thread = threading.Thread(target=lambda: rclpy.spin(ros_node), daemon=True)
    ros_thread.start()

    try:
        # Start the HTTP Server
        print(f"Starting web bridge server on port {port}")
        uvicorn.run(app, host="0.0.0.0", port=port)
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
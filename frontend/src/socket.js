import { io } from "socket.io-client";

// Connect to backend URL from env or fallback to localhost
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:5000";

export const socket = io(BACKEND_URL, {
  transports: ["websocket"],
  autoConnect: true,
});

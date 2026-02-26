import { useEffect, useRef, useState } from "react"
import { socket } from "../socket"

export default function Display() {
    const [frame, setFrame] = useState(null)
    const [fps, setFps] = useState(0)
    const frameCount = useRef(0)
    const lastFpsTime = useRef(Date.now())

    useEffect(() => {
        socket.on("frame", (data) => {
            setFrame(`data:image/jpeg;base64,${data.image}`)

            // Calculate actual received FPS
            frameCount.current++
            const now = Date.now()
            if (now - lastFpsTime.current >= 1000) {
                setFps(frameCount.current)
                frameCount.current = 0
                lastFpsTime.current = now
            }
        })

        return () => socket.off("frame")
    }, [])

    return (
        <div className="display-container">
            {frame ? (
                <>
                    <img
                        src={frame}
                        alt="Live camera feed from PC"
                        className="camera-feed"
                    />
                    <div className="fps-badge">{fps} fps</div>
                </>
            ) : (
                <div className="camera-placeholder">
                    <div className="placeholder-icon">📷</div>
                    <p>Waiting for camera feed...</p>
                    <p className="placeholder-sub">
                        Make sure the PC backend is running
                    </p>
                </div>
            )}
        </div>
    )
}

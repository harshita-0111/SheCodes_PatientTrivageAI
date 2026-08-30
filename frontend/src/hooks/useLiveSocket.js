import { useEffect, useRef, useState } from "react";

// PatientTriage.ai — live event feed hook.
//
// Connects to ws://<host>/ws/live and dispatches incoming messages by
// their `event` field. The backend has multiple independent
// broadcasters sharing one socket (new patient intake, nurse override,
// vitals update, and the background waiting-room monitor) — messages
// arrive interleaved, NOT as ordered request/response pairs. This hook
// exists specifically so no page has to re-implement that dispatch
// logic or make the (wrong) assumption that the next message on the
// socket is always the one it's waiting for.
//
// Reconnects automatically with a short backoff if the connection drops
// — a dev server restart or a laptop sleep/wake must not permanently
// kill the live feed.

function buildSocketUrl() {
  const customApiUrl = import.meta.env.VITE_API_URL;
  if (customApiUrl) {
    try {
      const url = new URL(customApiUrl);
      const protocol = url.protocol === "https:" ? "wss:" : "ws:";
      return `${protocol}//${url.host}/ws/live`;
    } catch (e) {
      console.error("Invalid VITE_API_URL for WebSocket:", e);
    }
  }

  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  // In dev, Vite's proxy (see vite.config.js) only forwards HTTP; for
  // the WebSocket we talk to the backend port directly.
  const host = import.meta.env.DEV ? "localhost:8000" : window.location.host;
  return `${protocol}//${host}/ws/live`;
}

export function useLiveSocket(handlers = {}) {
  const [connected, setConnected] = useState(false);
  const [lastEvent, setLastEvent] = useState(null);
  const handlersRef = useRef(handlers);
  handlersRef.current = handlers;

  useEffect(() => {
    let socket;
    let reconnectTimer;
    let cancelled = false;

    function connect() {
      socket = new WebSocket(buildSocketUrl());

      socket.onopen = () => !cancelled && setConnected(true);
      socket.onclose = () => {
        if (cancelled) return;
        setConnected(false);
        reconnectTimer = setTimeout(connect, 3000);
      };
      socket.onerror = () => socket.close();

      socket.onmessage = (raw) => {
        let data;
        try {
          data = JSON.parse(raw.data);
        } catch {
          return;
        }
        if (cancelled) return;
        setLastEvent(data);
        const handler = handlersRef.current[data.event];
        if (handler) handler(data);
      };
    }

    connect();
    return () => {
      cancelled = true;
      clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, []);

  return { connected, lastEvent };
}

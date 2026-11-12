import { useEffect } from "react";
import { useTerminalStore } from "../store/useTerminalStore";

export function useTerminalBootstrap() {
  const bootstrap = useTerminalStore((state) => state.bootstrap);
  const disconnectStreams = useTerminalStore((state) => state.disconnectStreams);

  useEffect(() => {
    bootstrap();
    return () => {
      disconnectStreams();
    };
  }, [bootstrap, disconnectStreams]);
}


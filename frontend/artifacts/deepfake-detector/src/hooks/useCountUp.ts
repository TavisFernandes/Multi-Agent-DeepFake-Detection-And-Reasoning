import { useState, useEffect } from 'react';

export function useCountUp(end: number, duration: number = 2000, decimals: number = 0, startTrigger: boolean = true) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    if (!startTrigger) return;
    
    let startTime: number | null = null;
    let animationFrame: number;

    const tick = (timestamp: number) => {
      if (!startTime) startTime = timestamp;
      const progress = timestamp - startTime;
      
      if (progress < duration) {
        // easeOutQuart
        const ease = 1 - Math.pow(1 - progress / duration, 4);
        setCount(end * ease);
        animationFrame = requestAnimationFrame(tick);
      } else {
        setCount(end);
      }
    };

    animationFrame = requestAnimationFrame(tick);
    
    return () => cancelAnimationFrame(animationFrame);
  }, [end, duration, startTrigger]);

  return count.toFixed(decimals);
}

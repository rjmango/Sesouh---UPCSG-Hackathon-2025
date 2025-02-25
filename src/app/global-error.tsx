'use client';

import Link from 'next/link';

interface GlobalErrorProps {
  error: Error & { digest?: string };
  reset: () => void;
}

export default function GlobalError({ error, reset }: GlobalErrorProps) {
  return (
    <main className="flex h-dvh w-dvw items-center justify-center">
      <h2>Something went wrong 😢</h2>
      <button onClick={() => reset()}>Try again</button>
    </main>
  );
}

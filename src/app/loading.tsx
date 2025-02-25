import { LoaderCircle } from 'lucide-react';

export default function Loading() {
  return (
    <div className="flex h-dvh w-dvw items-center justify-center">
      <LoaderCircle
        className="text-perx-red -ms-1 me-2 animate-spin"
        size={40}
        strokeWidth={2}
        aria-hidden="true"
      />
    </div>
  );
}

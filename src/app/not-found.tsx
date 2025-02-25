import Link from 'next/link';

export default function NotFound() {
  return (
    <main className="flex h-dvh w-dvw items-center justify-center">
      <h2>Page not found 😵</h2>
      <p>Requested resource could not be found</p>
      <Link href="/" className="hover:text-blue-800">
        Go to Home page
      </Link>
    </main>
  );
}

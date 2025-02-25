import Link from 'next/link';

export default function Sample() {
  return (
    <main>
      <div>Sample</div>
      <Link href="/" className="hover:text-blue-800">
        Return to Home
      </Link>
    </main>
  );
}

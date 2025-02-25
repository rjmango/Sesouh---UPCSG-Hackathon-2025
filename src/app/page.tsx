import Link from 'next/link';

export default function Home() {
  return (
    <main>
      <div className="bg-red-600 font-mono">Hello guyss</div>
      <Link href="/sample" className="hover:text-blue-800">
        Go to Sample page
      </Link>
    </main>
  );
}

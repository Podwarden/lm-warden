'use client';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';

export default function DonePage() {
  const router = useRouter();
  return (
    <section className="space-y-4">
      <h1 className="text-2xl font-semibold">Setup complete</h1>
      <p className="text-sm text-slate-400">
        LM Warden is ready. Log in with the admin account you just created.
      </p>
      <Button onClick={() => router.replace('/login')}>Go to LM Warden</Button>
    </section>
  );
}

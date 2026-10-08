// frontend/src/lib/api.ts
import type { paths } from './api-types.generated';
import { authFetch } from './auth-fetch';

type GET<P extends keyof paths> = paths[P] extends { get: { responses: { 200: { content: { 'application/json': infer R } } } } } ? R : never;
type POSTBody<P extends keyof paths> = paths[P] extends { post: { requestBody: { content: { 'application/json': infer B } } } } ? B : never;
type POSTResponse<P extends keyof paths> = paths[P] extends { post: { responses: { 200: { content: { 'application/json': infer R } } } } } ? R : paths[P] extends { post: { responses: { 201: { content: { 'application/json': infer R } } } } } ? R : void;

export async function getJSON<P extends keyof paths>(path: P): Promise<GET<P>> {
  const r = await authFetch(path as string);
  if (!r.ok) throw new Error(`${r.status} ${path as string}`);
  return r.json();
}

export async function postJSON<P extends keyof paths>(
  path: P, body: POSTBody<P>,
): Promise<POSTResponse<P>> {
  const r = await authFetch(path as string, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${path as string}`);
  return r.json();
}

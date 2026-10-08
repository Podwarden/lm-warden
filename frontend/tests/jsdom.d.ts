// `jsdom` ships no types of its own (see tests/setup.ts's localStorage fix,
// chat2 Task 12) and this repo does not otherwise depend on its API surface,
// so a full `@types/jsdom` devDependency isn't worth adding for one
// constructor call. Ambient `any` is enough for that one use site.
declare module 'jsdom';

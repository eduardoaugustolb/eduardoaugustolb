// Fixed theme variants for GitHub's explicit image-theme fragments.
// Originals remain the source of truth; run after regenerating ascii.svg.
const root = new URL('../', import.meta.url);
const portrait = await Bun.file(new URL('ascii.svg', root)).text();
const logo = await Bun.file(new URL('logo.svg', root)).text();
for (const theme of ['light', 'dark'] as const) {
  const dark = theme === 'dark';
  const portraitCSS = `.a{fill:${dark ? '#d8d0c3' : '#655d53'}}.ll{display:${dark ? 'none' : 'inline'}}.ld{display:${dark ? 'inline' : 'none'}}`;
  const fixed = portrait.replace(/\.a\{fill:#[\da-f]+\}\.ld\{display:none\}@media\(prefers-color-scheme:dark\)\{\.a\{fill:#[\da-f]+\}\.ll\{display:none\}\.ld\{display:inline\}\}/i, portraitCSS);
  if (fixed === portrait) throw new Error('Portrait theme rules changed; inspect before generating variants.');
  await Bun.write(new URL(`ascii-${theme}.svg`, root), fixed);
  const mark = logo.replace('width="390" height="294" viewBox="0 0 390 294"', 'width="286" height="190" viewBox="52 52 286 190"')
    .replace(/<style>.*?<\/style>/s, `<style>.mark{fill:${dark ? '#F7F3EA' : '#0B0B0B'}}</style>`);
  await Bun.write(new URL(`logo-${theme}.svg`, root), mark);
}

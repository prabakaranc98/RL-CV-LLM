// Layout check for the rendered deck: overflow into the footer zone or past the page edge,
// unrendered LaTeX, and broken images. Usage: node tools/check-layout.mjs dist/slides.html
import puppeteer from 'puppeteer-core';
import path from 'node:path';

const file = path.resolve(process.argv[2] || 'dist/slides.html');
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true, args: ['--allow-file-access-from-files'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1280, height: 720 });
await page.goto('file://' + file, { waitUntil: 'networkidle0' });
await page.evaluate(() => document.fonts.ready);

const problems = await page.evaluate(() => {
  const out = [];
  const sections = [...document.querySelectorAll('section[id]')];
  sections.forEach((sec, i) => {
    const n = i + 1;
    const r = sec.getBoundingClientRect();
    const scale = r.width / 1280;
    const hasFooter = !!sec.querySelector(':scope > footer');
    const limitY = hasFooter ? 720 - 64 : 720 - 24;           // content must end above the footer zone
    for (const el of sec.querySelectorAll(':scope > *:not(footer):not(header)')) {
      for (const node of [el, ...el.querySelectorAll('*')]) {
        const b = node.getBoundingClientRect();
        if (b.width === 0 || b.height === 0) continue;
        const bottom = (b.bottom - r.top) / scale, right = (b.right - r.left) / scale;
        if (bottom > limitY + 0.5) { out.push(`slide ${n}: <${node.tagName.toLowerCase()}> ends at y=${bottom.toFixed(0)} (limit ${limitY})`); break; }
        if (right > 1280 - 40) { out.push(`slide ${n}: <${node.tagName.toLowerCase()}> ends at x=${right.toFixed(0)}`); break; }
      }
    }
    const text = sec.innerText || '';
    if (/\$[^$\s][^$]*\$|\\(frac|pi|theta|mathrm|hat)\b/.test(text)) out.push(`slide ${n}: unrendered LaTeX in "${text.match(/.{0,30}(\$|\\)(frac|pi|theta|mathrm|hat)?.{0,30}/)?.[0]}"`);
    for (const img of sec.querySelectorAll('img')) if (!img.complete || img.naturalWidth === 0) out.push(`slide ${n}: broken image ${img.getAttribute('src')}`);
  });
  return { count: sections.length, out };
});
await browser.close();
console.log(`${problems.count} slides checked`);
if (problems.out.length) { console.log(problems.out.join('\n')); process.exit(1); }
console.log('layout check passed: no overflow, no unrendered LaTeX, no broken images');

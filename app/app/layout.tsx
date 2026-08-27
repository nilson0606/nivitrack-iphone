import type { Metadata, Viewport } from 'next';
import './globals.css';

const siteUrl = process.env.NEXT_PUBLIC_SITE_URL ?? 'http://localhost:3000/';
const previewUrl = new URL('og.png', siteUrl).href;

export const metadata: Metadata = {
  title: 'NiviTrack｜iPhone 本機影片追蹤',
  description: '在 iPhone 本機直接處理 MOV／HEVC，指定主角並完成 ViT 追蹤與影片輸出。',
  manifest: 'manifest.webmanifest',
  appleWebApp: {
    capable: true,
    title: 'NiviTrack',
    statusBarStyle: 'black-translucent',
  },
  openGraph: {
    title: 'NiviTrack',
    description: 'iPhone 本機影片追蹤',
    type: 'website',
    images: [{ url: previewUrl, width: 1200, height: 630, alt: 'NiviTrack iPhone 本機影片追蹤' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'NiviTrack',
    description: 'iPhone 本機影片追蹤',
    images: [previewUrl],
  },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  viewportFit: 'cover',
  themeColor: '#0a1410',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-Hant">
      <body>{children}</body>
    </html>
  );
}

import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'ToolCutter Studio',
  description: 'Segment tools with HQ-SAM and export SVG cutouts — powered by ToolCutter.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

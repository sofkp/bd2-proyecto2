import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TriModal Retrieval",
  description: "Buscador multimodal para texto, imagenes y audio",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="es" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}

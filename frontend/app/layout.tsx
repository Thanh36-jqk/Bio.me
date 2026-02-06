import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
    title: "Biometric MFA System",
    description: "Multi-Factor Authentication using Face, Iris, and Fingerprint Recognition with Deep Learning",
};

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en">
            <body className="antialiased">
                {children}
            </body>
        </html>
    );
}

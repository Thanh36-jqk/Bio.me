import Link from 'next/link';

export default function Home() {
    return (
        <main className="min-h-screen bg-[#0A0A0A]">
            {/* Subtle grid background */}
            <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:64px_64px]" />

            <div className="relative container mx-auto px-6 py-12 max-w-7xl">
                {/* Header - Minimalist */}
                <div className="mb-20">
                    <div className="flex items-center justify-between mb-12">
                        <div className="flex items-center gap-3">
                            <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
                            <span className="text-xs font-mono text-gray-500 uppercase tracking-wider">System Active</span>
                        </div>
                        <div className="flex gap-4 text-xs font-mono text-gray-600">
                            <span>v2.1.0</span>
                            <span className="text-gray-800">|</span>
                            <span>Enterprise</span>
                        </div>
                    </div>

                    <h1 className="text-7xl font-light tracking-tight text-white mb-6">
                        Biometric<br />
                        <span className="text-gray-600">Authentication</span>
                    </h1>
                    <p className="text-lg text-gray-500 max-w-2xl font-light">
                        Enterprise-grade multi-factor authentication powered by advanced neural networks
                    </p>
                </div>

                {/* Main Actions - Clean Cards */}
                <div className="grid md:grid-cols-2 gap-6 max-w-5xl mb-24">
                    <Link href="/login">
                        <div className="group relative bg-gradient-to-br from-zinc-900 to-zinc-950 border border-zinc-800 hover:border-zinc-700 rounded-xl p-8 transition-all duration-300 hover:shadow-2xl hover:shadow-emerald-500/5">
                            <div className="absolute top-4 right-4 text-xs font-mono text-zinc-700">01</div>
                            <div className="mb-6">
                                <svg className="w-8 h-8 text-emerald-500 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                                </svg>
                            </div>
                            <h2 className="text-2xl font-light text-white mb-3">Authenticate</h2>
                            <p className="text-sm text-gray-500 mb-6 leading-relaxed">
                                Three-layer biometric verification with real-time processing
                            </p>
                            <div className="flex gap-2 text-xs font-mono">
                                <span className="px-2 py-1 bg-zinc-800 text-emerald-500 rounded">Face</span>
                                <span className="px-2 py-1 bg-zinc-800 text-cyan-500 rounded">Iris</span>
                                <span className="px-2 py-1 bg-zinc-800 text-purple-500 rounded">Fingerprint</span>
                            </div>
                        </div>
                    </Link>

                    <Link href="/register">
                        <div className="group relative bg-gradient-to-br from-zinc-900 to-zinc-950 border border-zinc-800 hover:border-zinc-700 rounded-xl p-8 transition-all duration-300 hover:shadow-2xl hover:shadow-cyan-500/5">
                            <div className="absolute top-4 right-4 text-xs font-mono text-zinc-700">02</div>
                            <div className="mb-6">
                                <svg className="w-8 h-8 text-cyan-500 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z" />
                                </svg>
                            </div>
                            <h2 className="text-2xl font-light text-white mb-3">Register</h2>
                            <p className="text-sm text-gray-500 mb-6 leading-relaxed">
                                Secure enrollment with encrypted biometric template storage
                            </p>
                            <div className="flex gap-2 text-xs font-mono">
                                <span className="px-2 py-1 bg-zinc-800 text-zinc-500 rounded">AES-256</span>
                                <span className="px-2 py-1 bg-zinc-800 text-zinc-500 rounded">MongoDB</span>
                            </div>
                        </div>
                    </Link>
                </div>

                {/* Tech Stack - Professional Grid */}
                <div className="mb-24">
                    <div className="flex items-center gap-4 mb-8">
                        <h3 className="text-sm font-mono uppercase tracking-wider text-gray-600">Technology Stack</h3>
                        <div className="flex-1 h-px bg-gradient-to-r from-zinc-800 to-transparent" />
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="bg-zinc-950 border border-zinc-900 rounded-lg p-5 hover:border-zinc-800 transition-colors">
                            <div className="text-xs font-mono text-zinc-600 mb-2">FRONTEND</div>
                            <div className="text-lg font-light text-white mb-1">Next.js 14</div>
                            <div className="text-xs text-gray-600">React 18 • TypeScript</div>
                        </div>

                        <div className="bg-zinc-950 border border-zinc-900 rounded-lg p-5 hover:border-zinc-800 transition-colors">
                            <div className="text-xs font-mono text-zinc-600 mb-2">BACKEND</div>
                            <div className="text-lg font-light text-white mb-1">FastAPI</div>
                            <div className="text-xs text-gray-600">Python 3.11 • Uvicorn</div>
                        </div>

                        <div className="bg-zinc-950 border border-zinc-900 rounded-lg p-5 hover:border-zinc-800 transition-colors">
                            <div className="text-xs font-mono text-zinc-600 mb-2">AI/ML</div>
                            <div className="text-lg font-light text-white mb-1">InsightFace</div>
                            <div className="text-xs text-gray-600">ArcFace • ONNX Runtime</div>
                        </div>

                        <div className="bg-zinc-950 border border-zinc-900 rounded-lg p-5 hover:border-zinc-800 transition-colors">
                            <div className="text-xs font-mono text-zinc-600 mb-2">DATABASE</div>
                            <div className="text-lg font-light text-white mb-1">MongoDB</div>
                            <div className="text-xs text-gray-600">Atlas • Mongoose</div>
                        </div>
                    </div>
                </div>

                {/* Performance Metrics */}
                <div className="grid grid-cols-3 gap-6 max-w-4xl mb-24">
                    <div className="text-center">
                        <div className="text-4xl font-light text-emerald-500 mb-2">99.2%</div>
                        <div className="text-xs font-mono text-gray-600 uppercase tracking-wider">Face Accuracy</div>
                    </div>
                    <div className="text-center border-l border-r border-zinc-900">
                        <div className="text-4xl font-light text-cyan-500 mb-2">&lt;800ms</div>
                        <div className="text-xs font-mono text-gray-600 uppercase tracking-wider">Avg Response</div>
                    </div>
                    <div className="text-center">
                        <div className="text-4xl font-light text-purple-500 mb-2">256-bit</div>
                        <div className="text-xs font-mono text-gray-600 uppercase tracking-wider">Encryption</div>
                    </div>
                </div>

                {/* Footer */}
                <footer className="border-t border-zinc-900 pt-8">
                    <div className="flex items-center justify-between text-xs text-gray-700">
                        <div className="flex gap-6">
                            <a href="https://github.com/thanh36-jqk" target="_blank" rel="noopener noreferrer" className="hover:text-gray-500 transition-colors font-mono">
                                GitHub
                            </a>
                            <span className="font-mono">thanh36-jqk</span>
                        </div>
                        <div className="font-mono">
                            © 2026 Biometric MFA System
                        </div>
                    </div>
                </footer>
            </div>
        </main>
    );
}

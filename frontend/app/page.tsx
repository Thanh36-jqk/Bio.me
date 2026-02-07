'use client';

import Link from 'next/link';
import { useState, useEffect } from 'react';
import { Shield, Fingerprint, Eye, Scan, Lock, Zap, CheckCircle2, ArrowRight, Github, Mail, Server, Database, Cpu } from 'lucide-react';

export default function Home() {
    const [activeFeature, setActiveFeature] = useState(0);
    const [stats, setStats] = useState({ users: 0, scans: 0, accuracy: 0 });

    // Animated stats counter
    useEffect(() => {
        const interval = setInterval(() => {
            setStats(prev => ({
                users: Math.min(prev.users + 12, 1547),
                scans: Math.min(prev.scans + 87, 45289),
                accuracy: Math.min(prev.accuracy + 0.5, 99.2)
            }));
        }, 30);

        return () => clearInterval(interval);
    }, []);

    const features = [
        {
            icon: Scan,
            title: 'Face Recognition',
            description: 'Advanced neural networks for precise facial biometric authentication',
            tech: 'InsightFace + ArcFace',
            color: 'emerald'
        },
        {
            icon: Eye,
            title: 'Iris Scanning',
            description: 'Gabor filter encoding with Hamming distance matching',
            tech: 'OpenCV + Daugman Algorithm',
            color: 'cyan'
        },
        {
            icon: Fingerprint,
            title: 'Fingerprint Auth',
            description: 'ORB feature detection with adaptive threshold enhancement',
            tech: 'OpenCV ORB + BFMatcher',
            color: 'purple'
        }
    ];

    const securityFeatures = [
        { icon: Shield, label: 'Liveness Detection', description: 'Anti-spoofing technology' },
        { icon: Lock, label: 'AES-256 Encryption', description: 'Military-grade security' },
        { icon: Zap, label: 'Real-time Processing', description: 'Sub-second authentication' },
        { icon: CheckCircle2, label: '2/3 Pass Rule', description: 'Multi-factor verification' }
    ];

    const workflow = [
        { step: '01', title: 'Register', desc: 'Create account with email and biometric data' },
        { step: '02', title: 'Enroll', desc: 'Capture face, iris, and fingerprint samples' },
        { step: '03', title: 'Authenticate', desc: 'Login with 2 out of 3 biometric factors' },
        { step: '04', title: 'Access', desc: 'Secure access to protected resources' }
    ];

    return (
        <main className="min-h-screen bg-slate-900">
            {/* Gradient overlay */}
            <div className="fixed inset-0 bg-gradient-to-br from-slate-900 via-slate-900 to-slate-800 pointer-events-none" />

            {/* Grid pattern */}
            <div className="fixed inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:64px_64px] pointer-events-none" />

            <div className="relative">
                {/* Navigation */}
                <nav className="container mx-auto px-6 py-6 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                            <Shield className="text-white" size={20} />
                        </div>
                        <div>
                            <div className="text-white font-semibold">Bio.me</div>
                            <div className="text-xs text-slate-400">Biometric MFA</div>
                        </div>
                    </div>
                    <div className="flex gap-4">
                        <Link href="/login" className="px-4 py-2 text-slate-300 hover:text-white transition-colors text-sm">
                            Login
                        </Link>
                        <Link href="/register" className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors text-sm font-medium">
                            Get Started
                        </Link>
                    </div>
                </nav>

                {/* Hero Section */}
                <section className="container mx-auto px-6 py-20 text-center">
                    <div className="max-w-4xl mx-auto">
                        <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-500/10 border border-blue-500/20 rounded-full text-blue-400 text-sm mb-8">
                            <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                            Enterprise-Grade Security
                        </div>

                        <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight">
                            Advanced Biometric
                            <br />
                            <span className="bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                                Authentication
                            </span>
                        </h1>

                        <p className="text-xl text-slate-400 mb-12 max-w-2xl mx-auto">
                            Multi-factor biometric verification using face, iris, and fingerprint recognition
                            with AI-powered liveness detection
                        </p>

                        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
                            <Link href="/register" className="group px-8 py-4 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-all flex items-center gap-2 font-medium shadow-lg shadow-blue-600/20">
                                Start Free Trial
                                <ArrowRight className="group-hover:translate-x-1 transition-transform" size={18} />
                            </Link>
                            <Link href="/login" className="px-8 py-4 bg-slate-800 hover:bg-slate-700 text-white rounded-lg transition-colors font-medium border border-slate-700">
                                Sign In
                            </Link>
                        </div>

                        {/* Live Stats */}
                        <div className="grid grid-cols-3 gap-8 mt-16 max-w-2xl mx-auto">
                            <div>
                                <div className="text-3xl font-bold text-white mb-1">{stats.users.toLocaleString()}</div>
                                <div className="text-sm text-slate-400">Active Users</div>
                            </div>
                            <div>
                                <div className="text-3xl font-bold text-white mb-1">{stats.scans.toLocaleString()}</div>
                                <div className="text-sm text-slate-400">Auth Scans</div>
                            </div>
                            <div>
                                <div className="text-3xl font-bold text-white mb-1">{stats.accuracy.toFixed(1)}%</div>
                                <div className="text-sm text-slate-400">Accuracy</div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Features Section */}
                <section className="container mx-auto px-6 py-20">
                    <div className="text-center mb-16">
                        <h2 className="text-4xl font-bold text-white mb-4">Three-Layer Protection</h2>
                        <p className="text-slate-400 text-lg">Advanced AI algorithms for uncompromising security</p>
                    </div>

                    <div className="grid md:grid-cols-3 gap-6">
                        {features.map((feature, idx) => {
                            const Icon = feature.icon;
                            return (
                                <div
                                    key={idx}
                                    onClick={() => setActiveFeature(idx)}
                                    className={`
                                        bg-slate-800 border rounded-xl p-8 transition-all cursor-pointer
                                        ${activeFeature === idx
                                            ? 'border-blue-500 shadow-lg shadow-blue-500/20 scale-105'
                                            : 'border-slate-700 hover:border-slate-600'
                                        }
                                    `}
                                >
                                    <div className={`w-14 h-14 bg-${feature.color}-500/10 rounded-lg flex items-center justify-center mb-6`}>
                                        <Icon className={`text-${feature.color}-400`} size={28} />
                                    </div>
                                    <h3 className="text-2xl font-semibold text-white mb-3">{feature.title}</h3>
                                    <p className="text-slate-400 mb-4">{feature.description}</p>
                                    <div className="inline-block px-3 py-1 bg-slate-700 rounded text-xs text-slate-300 font-mono">
                                        {feature.tech}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </section>

                {/* How It Works */}
                <section className="container mx-auto px-6 py-20">
                    <div className="text-center mb-16">
                        <h2 className="text-4xl font-bold text-white mb-4">How It Works</h2>
                        <p className="text-slate-400 text-lg">Simple, secure, and seamless authentication</p>
                    </div>

                    <div className="grid md:grid-cols-4 gap-8 max-w-6xl mx-auto">
                        {workflow.map((item, idx) => (
                            <div key={idx} className="relative">
                                {idx < workflow.length - 1 && (
                                    <div className="hidden md:block absolute top-8 left-full w-full h-0.5 bg-slate-700" />
                                )}
                                <div className="relative">
                                    <div className="w-16 h-16 bg-blue-600 rounded-full flex items-center justify-center text-white font-bold text-xl mb-4 mx-auto shadow-lg shadow-blue-600/30">
                                        {item.step}
                                    </div>
                                    <h3 className="text-xl font-semibold text-white mb-2 text-center">{item.title}</h3>
                                    <p className="text-slate-400 text-sm text-center">{item.desc}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </section>

                {/* Security Features */}
                <section className="container mx-auto px-6 py-20">
                    <div className="bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-12">
                        <div className="text-center mb-12">
                            <h2 className="text-4xl font-bold text-white mb-4">Enterprise Security</h2>
                            <p className="text-slate-400 text-lg">Military-grade protection for your data</p>
                        </div>

                        <div className="grid md:grid-cols-4 gap-6">
                            {securityFeatures.map((item, idx) => {
                                const Icon = item.icon;
                                return (
                                    <div key={idx} className="text-center">
                                        <div className="w-16 h-16 bg-blue-500/10 rounded-lg flex items-center justify-center mb-4 mx-auto">
                                            <Icon className="text-blue-400" size={28} />
                                        </div>
                                        <h3 className="text-white font-semibold mb-2">{item.label}</h3>
                                        <p className="text-slate-400 text-sm">{item.description}</p>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                </section>

                {/* Tech Stack */}
                <section className="container mx-auto px-6 py-20">
                    <div className="text-center mb-16">
                        <h2 className="text-4xl font-bold text-white mb-4">Technology Stack</h2>
                        <p className="text-slate-400 text-lg">Built with cutting-edge technologies</p>
                    </div>

                    <div className="grid md:grid-cols-4 gap-6 max-w-5xl mx-auto">
                        <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 hover:border-blue-500 transition-colors">
                            <Server className="text-blue-400 mb-4" size={32} />
                            <div className="text-xs text-slate-500 mb-2 uppercase tracking-wider">Frontend</div>
                            <div className="text-xl font-semibold text-white mb-1">Next.js 14</div>
                            <div className="text-sm text-slate-400">React • TypeScript</div>
                        </div>

                        <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 hover:border-blue-500 transition-colors">
                            <Cpu className="text-cyan-400 mb-4" size={32} />
                            <div className="text-xs text-slate-500 mb-2 uppercase tracking-wider">Backend</div>
                            <div className="text-xl font-semibold text-white mb-1">FastAPI</div>
                            <div className="text-sm text-slate-400">Python • Uvicorn</div>
                        </div>

                        <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 hover:border-blue-500 transition-colors">
                            <Database className="text-purple-400 mb-4" size={32} />
                            <div className="text-xs text-slate-500 mb-2 uppercase tracking-wider">Database</div>
                            <div className="text-xl font-semibold text-white mb-1">MongoDB</div>
                            <div className="text-sm text-slate-400">Atlas • Cloud</div>
                        </div>

                        <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 hover:border-blue-500 transition-colors">
                            <Shield className="text-emerald-400 mb-4" size={32} />
                            <div className="text-xs text-slate-500 mb-2 uppercase tracking-wider">AI/ML</div>
                            <div className="text-xl font-semibold text-white mb-1">OpenCV</div>
                            <div className="text-sm text-slate-400">InsightFace • ORB</div>
                        </div>
                    </div>
                </section>

                {/* CTA Section */}
                <section className="container mx-auto px-6 py-20">
                    <div className="bg-gradient-to-r from-blue-600 to-cyan-600 rounded-2xl p-12 text-center">
                        <h2 className="text-4xl font-bold text-white mb-4">Ready to Get Started?</h2>
                        <p className="text-blue-100 text-lg mb-8 max-w-2xl mx-auto">
                            Join thousands of users securing their applications with advanced biometric authentication
                        </p>
                        <div className="flex flex-col sm:flex-row gap-4 justify-center">
                            <Link href="/register" className="px-8 py-4 bg-white text-blue-600 rounded-lg font-semibold hover:bg-blue-50 transition-colors shadow-lg">
                                Create Free Account
                            </Link>
                            <a href="https://github.com/thanh36-jqk" target="_blank" rel="noopener noreferrer" className="px-8 py-4 bg-blue-700 text-white rounded-lg font-semibold hover:bg-blue-800 transition-colors flex items-center justify-center gap-2">
                                <Github size={20} />
                                View on GitHub
                            </a>
                        </div>
                    </div>
                </section>

                {/* Footer */}
                <footer className="border-t border-slate-800">
                    <div className="container mx-auto px-6 py-12">
                        <div className="grid md:grid-cols-4 gap-8 mb-8">
                            <div>
                                <div className="flex items-center gap-3 mb-4">
                                    <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                                        <Shield className="text-white" size={20} />
                                    </div>
                                    <div>
                                        <div className="text-white font-semibold">Bio.me</div>
                                        <div className="text-xs text-slate-400">Biometric MFA</div>
                                    </div>
                                </div>
                                <p className="text-slate-400 text-sm">
                                    Enterprise-grade multi-factor biometric authentication system.
                                </p>
                            </div>

                            <div>
                                <h3 className="text-white font-semibold mb-4">Product</h3>
                                <ul className="space-y-2 text-slate-400 text-sm">
                                    <li><Link href="/register" className="hover:text-white transition-colors">Register</Link></li>
                                    <li><Link href="/login" className="hover:text-white transition-colors">Login</Link></li>
                                    <li><a href="#features" className="hover:text-white transition-colors">Features</a></li>
                                </ul>
                            </div>

                            <div>
                                <h3 className="text-white font-semibold mb-4">Technology</h3>
                                <ul className="space-y-2 text-slate-400 text-sm">
                                    <li>Face Recognition</li>
                                    <li>Iris Scanning</li>
                                    <li>Fingerprint Auth</li>
                                </ul>
                            </div>

                            <div>
                                <h3 className="text-white font-semibold mb-4">Connect</h3>
                                <div className="flex gap-4">
                                    <a href="https://github.com/thanh36-jqk" target="_blank" rel="noopener noreferrer" className="w-10 h-10 bg-slate-800 hover:bg-slate-700 rounded-lg flex items-center justify-center transition-colors">
                                        <Github className="text-slate-400" size={20} />
                                    </a>
                                    <a href="mailto:thanh36@example.com" className="w-10 h-10 bg-slate-800 hover:bg-slate-700 rounded-lg flex items-center justify-center transition-colors">
                                        <Mail className="text-slate-400" size={20} />
                                    </a>
                                </div>
                            </div>
                        </div>

                        <div className="border-t border-slate-800 pt-8 flex flex-col md:flex-row justify-between items-center text-sm text-slate-400">
                            <div>© 2026 Bio.me. All rights reserved.</div>
                            <div className="flex gap-6 mt-4 md:mt-0">
                                <a href="#" className="hover:text-white transition-colors">Privacy Policy</a>
                                <a href="#" className="hover:text-white transition-colors">Terms of Service</a>
                                <a href="#" className="hover:text-white transition-colors">Documentation</a>
                            </div>
                        </div>
                    </div>
                </footer>
            </div>
        </main>
    );
}

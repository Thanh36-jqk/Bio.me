'use client';

import { useState } from 'react';
import Link from 'next/link';
import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface BiometricResult {
    success: boolean;
    message: string;
    confidence?: number;
    distance?: number;
}

interface ResultsState {
    face: BiometricResult | null;
    iris: BiometricResult | null;
    fingerprint: BiometricResult | null;
}

export default function LoginPage() {
    const [step, setStep] = useState(0); // 0: username, 1: face, 2: iris, 3: fingerprint, 4: success
    const [username, setUsername] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [results, setResults] = useState<ResultsState>({ face: null, iris: null, fingerprint: null });

    const handleUsernameSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        setLoading(true);

        try {
            const response = await axios.get(`${API_BASE}/api/user/${username}`);
            if (response.data.exists) {
                setStep(1);
            } else {
                setError('User not found. Please register first.');
            }
        } catch (err) {
            setError('Error checking user. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const handleFileUpload = async (file: File, type: 'face' | 'iris' | 'fingerprint') => {
        setError('');
        setLoading(true);

        try {
            const formData = new FormData();
            formData.append('username', username);
            formData.append('file', file);

            const response = await axios.post(`${API_BASE}/api/${type}/verify`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });

            setResults(prev => ({ ...prev, [type]: response.data }));

            if (response.data.success) {
                // Move to next step
                if (type === 'face') setStep(2);
                else if (type === 'iris') setStep(3);
                else if (type === 'fingerprint') setStep(4);
            } else {
                setError(`${type.charAt(0).toUpperCase() + type.slice(1)} verification failed. Please try again.`);
            }
        } catch (err: any) {
            setError(err.response?.data?.detail || `Error during ${type} verification`);
        } finally {
            setLoading(false);
        }
    };

    return (
        <main className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-indigo-900 py-12 px-4">
            <div className="max-w-2xl mx-auto">
                {/* Header */}
                <div className="text-center mb-8">
                    <Link href="/" className="text-blue-400 hover:text-blue-300 text-sm">← Back to Home</Link>
                    <h1 className="text-4xl font-bold mt-4 text-white">Biometric Login</h1>
                    <p className="text-gray-300 mt-2">Three-factor authentication</p>
                </div>

                {/* Progress Bar */}
                <div className="mb-8">
                    <div className="flex justify-between items-center">
                        {['Username', 'Face', 'Iris', 'Fingerprint'].map((label, idx) => (
                            <div key={idx} className="flex flex-col items-center">
                                <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold ${step > idx ? 'bg-green-500' : step === idx ? 'bg-blue-500' : 'bg-gray-600'
                                    }`}>
                                    {step > idx ? '✓' : idx + 1}
                                </div>
                                <span className="text-xs mt-1 text-gray-400">{label}</span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Main Card */}
                <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20">
                    {error && (
                        <div className="bg-red-500/20 border border-red-500/50 text-red-200 px-4 py-3 rounded mb-4">
                            {error}
                        </div>
                    )}

                    {/* Step 0: Username */}
                    {step === 0 && (
                        <form onSubmit={handleUsernameSubmit}>
                            <label className="block text-white font-bold mb-2">Username</label>
                            <input
                                type="text"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                                className="w-full px-4 py-3 rounded-lg bg-white/10 border border-white/30 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                placeholder="Enter your username"
                                required
                            />
                            <button
                                type="submit"
                                disabled={loading}
                                className="w-full mt-4 bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                            >
                                {loading ? 'Checking...' : 'Continue →'}
                            </button>
                        </form>
                    )}

                    {/* Step 1: Face */}
                    {step === 1 && (
                        <div>
                            <h3 className="text-xl font-bold text-white mb-4">Step 1: Face Verification</h3>
                            <p className="text-gray-300 mb-4">Upload a clear photo of your face</p>
                            <input
                                type="file"
                                accept="image/*"
                                onChange={(e) => e.target.files && handleFileUpload(e.target.files[0], 'face')}
                                className="w-full text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:bg-blue-600 file:text-white hover:file:bg-blue-700"
                                disabled={loading}
                            />
                            {loading && <p className="mt-4 text-center text-blue-400">Processing...</p>}
                        </div>
                    )}

                    {/* Step 2: Iris */}
                    {step === 2 && (
                        <div>
                            <h3 className="text-xl font-bold text-white mb-4">Step 2: Iris Verification</h3>
                            <p className="text-gray-300 mb-4">Upload a clear eye/iris image</p>
                            <input
                                type="file"
                                accept="image/*"
                                onChange={(e) => e.target.files && handleFileUpload(e.target.files[0], 'iris')}
                                className="w-full text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:bg-green-600 file:text-white hover:file:bg-green-700"
                                disabled={loading}
                            />
                            {loading && <p className="mt-4 text-center text-green-400">Processing...</p>}
                        </div>
                    )}

                    {/* Step 3: Fingerprint */}
                    {step === 3 && (
                        <div>
                            <h3 className="text-xl font-bold text-white mb-4">Step 3: Fingerprint Verification</h3>
                            <p className="text-gray-300 mb-4">Upload a fingerprint image</p>
                            <input
                                type="file"
                                accept="image/*"
                                onChange={(e) => e.target.files && handleFileUpload(e.target.files[0], 'fingerprint')}
                                className="w-full text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:bg-purple-600 file:text-white hover:file:bg-purple-700"
                                disabled={loading}
                            />
                            {loading && <p className="mt-4 text-center text-purple-400">Processing...</p>}
                        </div>
                    )}

                    {/* Step 4: Success */}
                    {step === 4 && (
                        <div className="text-center">
                            <div className="text-6xl mb-4">🎉</div>
                            <h2 className="text-3xl font-bold text-green-400 mb-2">Authentication Successful!</h2>
                            <p className="text-gray-300 mb-6">Welcome back, <strong>{username}</strong></p>

                            <div className="grid grid-cols-3 gap-4 mb-6">
                                <div className="bg-green-500/20 rounded-lg p-4">
                                    <div className="text-2xl mb-2">✓</div>
                                    <div className="text-sm">Face</div>
                                    <div className="text-xs text-gray-400 mt-1">
                                        {results.face?.confidence ? `${(results.face.confidence * 100).toFixed(1)}%` : 'Verified'}
                                    </div>
                                </div>
                                <div className="bg-green-500/20 rounded-lg p-4">
                                    <div className="text-2xl mb-2">✓</div>
                                    <div className="text-sm">Iris</div>
                                    <div className="text-xs text-gray-400 mt-1">
                                        {results.iris?.distance ? `${(results.iris.distance * 100).toFixed(1)}%` : 'Verified'}
                                    </div>
                                </div>
                                <div className="bg-green-500/20 rounded-lg p-4">
                                    <div className="text-2xl mb-2">✓</div>
                                    <div className="text-sm">Fingerprint</div>
                                    <div className="text-xs text-gray-400 mt-1">
                                        {results.fingerprint?.confidence ? `${(results.fingerprint.confidence * 100).toFixed(1)}%` : 'Verified'}
                                    </div>
                                </div>
                            </div>

                            <Link href="/">
                                <button className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-8 rounded-lg transition-colors">
                                    Return to Home
                                </button>
                            </Link>
                        </div>
                    )}
                </div>

                {/* Info */}
                {step > 0 && step < 4 && (
                    <div className="mt-6 text-center text-sm text-gray-400">
                        <p>All biometric data is encrypted and securely stored</p>
                    </div>
                )}
            </div>
        </main>
    );
}

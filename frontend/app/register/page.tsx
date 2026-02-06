'use client';

import { useState } from 'react';
import Link from 'next/link';
import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function RegisterPage() {
    const [step, setStep] = useState(0); // 0: username, 1: face, 2: iris, 3: fingerprint, 4: success
    const [username, setUsername] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [faceFiles, setFaceFiles] = useState<File[]>([]);
    const [irisFiles, setIrisFiles] = useState<File[]>([]);
    const [fpFiles, setFpFiles] = useState<File[]>([]);

    const handleUsernameSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');

        if (username.length < 3) {
            setError('Username must be at least 3 characters');
            return;
        }

        // Skip API call - backend creates user on first biometric upload
        setStep(1);
    };

    const handleBiometricUpload = async (files: File[], type: 'face' | 'iris' | 'fingerprint') => {
        if (files.length < 3) {
            setError(`Please upload at least 3 ${type} images`);
            return;
        }

        setError('');
        setLoading(true);

        try {
            const formData = new FormData();
            formData.append('username', username);
            files.forEach(file => formData.append('files', file));

            const response = await axios.post(`${API_BASE}/register/${type}`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });

            if (response.data.success) {
                if (type === 'face') setStep(2);
                else if (type === 'iris') setStep(3);
                else if (type === 'fingerprint') setStep(4);
            }
        } catch (err: any) {
            setError(err.response?.data?.detail || `Error during ${type} registration`);
        } finally {
            setLoading(false);
        }
    };

    return (
        <main className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-pink-900 py-12 px-4">
            <div className="max-w-2xl mx-auto">
                {/* Header */}
                <div className="text-center mb-8">
                    <Link href="/" className="text-purple-400 hover:text-purple-300 text-sm">← Back to Home</Link>
                    <h1 className="text-4xl font-bold mt-4 text-white">Create Account</h1>
                    <p className="text-gray-300 mt-2">Register your biometric credentials</p>
                </div>

                {/* Progress */}
                <div className="mb-8">
                    <div className="flex justify-between">
                        {['Username', 'Face', 'Iris', 'Fingerprint'].map((label, idx) => (
                            <div key={idx} className={`flex-1 ${idx < 3 ? 'border-r-2' : ''} ${step > idx ? 'border-green-500' : 'border-gray-600'} pb-2`}>
                                <div className={`text-center font-bold ${step >= idx ? 'text-white' : 'text-gray-600'}`}>
                                    {step > idx ? '✓' : idx + 1}. {label}
                                </div>
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
                            <label className="block text-white font-bold mb-2">Choose a Username</label>
                            <input
                                type="text"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                                className="w-full px-4 py-3 rounded-lg bg-white/10 border border-white/30 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                                placeholder="Username"
                                required
                                minLength={3}
                            />
                            <p className="text-sm text-gray-400 mt-2">Minimum 3 characters</p>
                            <button
                                type="submit"
                                disabled={loading}
                                className="w-full mt-4 bg-purple-600 hover:bg-purple-700 text-white font-bold py-3 px-6 rounded-lg disabled:opacity-50 transition-colors"
                            >
                                {loading ? 'Creating...' : 'Continue →'}
                            </button>
                        </form>
                    )}

                    {/* Step 1: Face */}
                    {step === 1 && (
                        <div>
                            <h3 className="text-xl font-bold text-white mb-4">Step 1: Face Registration</h3>
                            <p className="text-gray-300 mb-4">Upload 5-15 clear photos of your face from different angles</p>
                            <input
                                type="file"
                                accept="image/*"
                                multiple
                                onChange={(e) => setFaceFiles(Array.from(e.target.files || []))}
                                className="w-full text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:bg-blue-600 file:text-white hover:file:bg-blue-700"
                            />
                            {faceFiles.length > 0 && (
                                <p className="mt-2 text-sm text-green-400">{faceFiles.length} files selected</p>
                            )}
                            <button
                                onClick={() => handleBiometricUpload(faceFiles, 'face')}
                                disabled={loading || faceFiles.length < 3}
                                className="w-full mt-4 bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg disabled:opacity-50 transition-colors"
                            >
                                {loading ? 'Processing...' : `Register Face (${faceFiles.length}/15)`}
                            </button>
                        </div>
                    )}

                    {/* Step 2: Iris */}
                    {step === 2 && (
                        <div>
                            <h3 className="text-xl font-bold text-white mb-4">Step 2: Iris Registration</h3>
                            <p className="text-gray-300 mb-4">Upload 3-10 clear iris/eye images</p>
                            <input
                                type="file"
                                accept="image/*"
                                multiple
                                onChange={(e) => setIrisFiles(Array.from(e.target.files || []))}
                                className="w-full text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:bg-green-600 file:text-white hover:file:bg-green-700"
                            />
                            {irisFiles.length > 0 && (
                                <p className="mt-2 text-sm text-green-400">{irisFiles.length} files selected</p>
                            )}
                            <button
                                onClick={() => handleBiometricUpload(irisFiles, 'iris')}
                                disabled={loading || irisFiles.length < 3}
                                className="w-full mt-4 bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-lg disabled:opacity-50 transition-colors"
                            >
                                {loading ? 'Processing...' : `Register Iris (${irisFiles.length}/10)`}
                            </button>
                        </div>
                    )}

                    {/* Step 3: Fingerprint */}
                    {step === 3 && (
                        <div>
                            <h3 className="text-xl font-bold text-white mb-4">Step 3: Fingerprint Registration</h3>
                            <p className="text-gray-300 mb-4">Upload 3-10 fingerprint images</p>
                            <input
                                type="file"
                                accept="image/*"
                                multiple
                                onChange={(e) => setFpFiles(Array.from(e.target.files || []))}
                                className="w-full text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:bg-purple-600 file:text-white hover:file:bg-purple-700"
                            />
                            {fpFiles.length > 0 && (
                                <p className="mt-2 text-sm text-green-400">{fpFiles.length} files selected</p>
                            )}
                            <button
                                onClick={() => handleBiometricUpload(fpFiles, 'fingerprint')}
                                disabled={loading || fpFiles.length < 3}
                                className="w-full mt-4 bg-purple-600 hover:bg-purple-700 text-white font-bold py-3 px-6 rounded-lg disabled:opacity-50 transition-colors"
                            >
                                {loading ? 'Processing...' : `Register Fingerprint (${fpFiles.length}/10)`}
                            </button>
                        </div>
                    )}

                    {/* Step 4: Success */}
                    {step === 4 && (
                        <div className="text-center">
                            <div className="text-6xl mb-4">🎊</div>
                            <h2 className="text-3xl font-bold text-green-400 mb-2">Registration Complete!</h2>
                            <p className="text-gray-300 mb-6">Your account <strong>{username}</strong> has been created successfully.</p>
                            <Link href="/login">
                                <button className="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-8 rounded-lg transition-colors">
                                    Go to Login →
                                </button>
                            </Link>
                        </div>
                    )}
                </div>
            </div>
        </main>
    );
}

'use client';

import { useState, useRef, useCallback } from 'react';
import Link from 'next/link';
import axios from 'axios';
import Webcam from 'react-webcam';
import { Camera, Upload, Check, X, Mail, ShieldCheck, ShieldAlert } from 'lucide-react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function LoginPage() {
    const [step, setStep] = useState(0); // 0: email, 1: biometrics, 2: processing, 3: result
    const [email, setEmail] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    // Biometric Data
    const webcamRef = useRef<Webcam>(null);
    const [faceImage, setFaceImage] = useState<Blob | null>(null);
    const [facePreview, setFacePreview] = useState<string | null>(null);

    const [irisFile, setIrisFile] = useState<File | null>(null);
    const [fingerprintFile, setFingerprintFile] = useState<File | null>(null);

    // Auth Result
    const [authResult, setAuthResult] = useState<{
        success: boolean;
        message: string;
        passed_biometrics: number;
        liveness_checks?: any;
    } | null>(null);

    const handleEmailSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');

        if (!email.includes('@')) {
            setError('Please enter a valid email');
            return;
        }

        // Skip user check API call for now to avoid complexity, 
        // rely on main auth endpoint to check user existence
        setStep(1);
    };

    const captureFace = useCallback(() => {
        if (webcamRef.current) {
            const imageSrc = webcamRef.current.getScreenshot();
            if (imageSrc) {
                setFacePreview(imageSrc);
                // Convert base64 to blob
                fetch(imageSrc)
                    .then(res => res.blob())
                    .then(blob => setFaceImage(blob));
            }
        }
    }, [webcamRef]);

    const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>, type: 'iris' | 'fingerprint') => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            if (type === 'iris') setIrisFile(file);
            else setFingerprintFile(file);
        }
    };

    const handleLogin = async () => {
        // Count provided biometrics
        let count = 0;
        if (faceImage) count++;
        if (irisFile) count++;
        if (fingerprintFile) count++;

        if (count < 2) {
            setError('Please provide at least 2 biometrics to login (2/3 Rule)');
            return;
        }

        setLoading(true);
        setError('');
        setStep(2); // Processing

        try {
            const formData = new FormData();
            formData.append('email', email);

            if (faceImage) formData.append('face_image', faceImage, 'face.jpg');
            if (irisFile) formData.append('iris_image', irisFile);
            if (fingerprintFile) formData.append('fingerprint_image', fingerprintFile);

            const response = await axios.post(`${API_BASE}/authenticate`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });

            setAuthResult(response.data);
            setStep(3); // Result
        } catch (err: any) {
            setError(err.response?.data?.message || 'Authentication failed. Please try again.');
            setStep(1);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen relative overflow-hidden bg-[#0A0A0A] text-white selection:bg-purple-500/30">
            {/* Background Effects */}
            <div className="absolute top-0 left-0 w-full h-full overflow-hidden -z-10">
                <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-blue-600/10 blur-[120px]" />
                <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] rounded-full bg-purple-600/10 blur-[120px]" />
            </div>

            <div className="container mx-auto px-4 py-8">
                <div className="mb-8">
                    <Link href="/" className="text-gray-400 hover:text-white transition-colors flex items-center gap-2">
                        ← Back to Home
                    </Link>
                </div>

                <div className="max-w-xl mx-auto">
                    <div className="text-center mb-10">
                        <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400 mb-4">
                            Biometric Login
                        </h1>
                        <p className="text-gray-400 text-lg">
                            Secure 2/3 Multi-Factor Authentication
                        </p>
                    </div>

                    <div className="bg-gray-900/50 backdrop-blur-xl border border-white/5 rounded-2xl p-8 shadow-2xl">
                        {error && (
                            <div className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 flex items-center gap-2">
                                <ShieldAlert size={18} />
                                {error}
                            </div>
                        )}

                        {/* Step 0: Email Input */}
                        {step === 0 && (
                            <form onSubmit={handleEmailSubmit} className="space-y-6">
                                <div>
                                    <label className="block text-sm font-medium text-gray-400 mb-2">Email Address</label>
                                    <div className="relative">
                                        <Mail className="absolute left-4 top-3.5 text-gray-500" size={20} />
                                        <input
                                            type="email"
                                            value={email}
                                            onChange={(e) => setEmail(e.target.value)}
                                            className="w-full bg-gray-950 border border-gray-800 rounded-xl py-3 pl-12 pr-4 text-white focus:outline-none focus:border-blue-500 transition-colors"
                                            placeholder="john@example.com"
                                            required
                                        />
                                    </div>
                                </div>
                                <button
                                    type="submit"
                                    className="w-full bg-blue-600 hover:bg-blue-500 text-white font-medium py-4 rounded-xl transition-all"
                                >
                                    Continue →
                                </button>
                            </form>
                        )}

                        {/* Step 1: Biometric Input */}
                        {step === 1 && (
                            <div className="space-y-8">
                                <p className="text-center text-gray-400 text-sm">
                                    Provide at least 2 biometrics to verify identity
                                </p>

                                {/* 1. Face (Webcam) */}
                                <div className={`border rounded-xl p-4 transition-all ${faceImage ? 'border-green-500/50 bg-green-500/5' : 'border-gray-800 bg-gray-950'}`}>
                                    <div className="flex justify-between items-center mb-4">
                                        <h3 className="font-medium flex items-center gap-2">
                                            <Camera className="text-blue-400" size={18} /> Face Verification
                                        </h3>
                                        {faceImage && <Check className="text-green-500" size={18} />}
                                    </div>

                                    {!facePreview ? (
                                        <div className="space-y-4">
                                            <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
                                                <Webcam
                                                    audio={false}
                                                    ref={webcamRef}
                                                    screenshotFormat="image/jpeg"
                                                    className="w-full h-full object-cover"
                                                />
                                            </div>
                                            <button
                                                onClick={captureFace}
                                                className="w-full py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm transition-colors"
                                            >
                                                Capture Face
                                            </button>
                                        </div>
                                    ) : (
                                        <div className="flex gap-4">
                                            <img src={facePreview} alt="Face" className="w-20 h-20 rounded-lg object-cover border border-gray-700" />
                                            <button
                                                onClick={() => { setFacePreview(null); setFaceImage(null); }}
                                                className="text-sm text-gray-400 hover:text-white underline"
                                            >
                                                Retake
                                            </button>
                                        </div>
                                    )}
                                </div>

                                {/* 2. Iris (Upload) */}
                                <div className={`border rounded-xl p-4 transition-all ${irisFile ? 'border-green-500/50 bg-green-500/5' : 'border-gray-800 bg-gray-950'}`}>
                                    <div className="flex justify-between items-center mb-4">
                                        <h3 className="font-medium flex items-center gap-2">
                                            <Upload className="text-purple-400" size={18} /> Iris Upload
                                        </h3>
                                        {irisFile && <Check className="text-green-500" size={18} />}
                                    </div>
                                    <input
                                        type="file"
                                        accept="image/*"
                                        onChange={(e) => handleFileUpload(e, 'iris')}
                                        className="text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-gray-800 file:text-white hover:file:bg-gray-700"
                                    />
                                </div>

                                {/* 3. Fingerprint (Upload) */}
                                <div className={`border rounded-xl p-4 transition-all ${fingerprintFile ? 'border-green-500/50 bg-green-500/5' : 'border-gray-800 bg-gray-950'}`}>
                                    <div className="flex justify-between items-center mb-4">
                                        <h3 className="font-medium flex items-center gap-2">
                                            <Upload className="text-orange-400" size={18} /> Fingerprint Upload
                                        </h3>
                                        {fingerprintFile && <Check className="text-green-500" size={18} />}
                                    </div>
                                    <input
                                        type="file"
                                        accept="image/*"
                                        onChange={(e) => handleFileUpload(e, 'fingerprint')}
                                        className="text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-gray-800 file:text-white hover:file:bg-gray-700"
                                    />
                                </div>

                                <button
                                    onClick={handleLogin}
                                    disabled={loading}
                                    className="w-full bg-blue-600 hover:bg-blue-500 text-white font-medium py-4 rounded-xl transition-all disabled:opacity-50"
                                >
                                    Verify & Login
                                </button>
                            </div>
                        )}

                        {/* Step 2: Processing */}
                        {step === 2 && (
                            <div className="text-center py-12">
                                <div className="animate-spin w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-6"></div>
                                <h3 className="text-xl font-medium text-white">Verifying Identity...</h3>
                                <p className="text-gray-400 mt-2">Checking liveness and biometrics</p>
                            </div>
                        )}

                        {/* Step 3: Result */}
                        {step === 3 && authResult && (
                            <div className="text-center py-8">
                                <div className={`w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-6 ${authResult.success ? 'bg-green-500/20 text-green-500' : 'bg-red-500/20 text-red-500'}`}>
                                    {authResult.success ? <ShieldCheck size={40} /> : <ShieldAlert size={40} />}
                                </div>

                                <h2 className="text-2xl font-bold text-white mb-2">
                                    {authResult.success ? 'Access Granted' : 'Access Denied'}
                                </h2>
                                <p className="text-gray-400 mb-6">{authResult.message}</p>

                                <div className="bg-gray-950 rounded-xl p-4 mb-6 text-left">
                                    <h4 className="text-sm font-medium text-gray-500 mb-3 uppercase tracking-wider">Analysis Report</h4>
                                    <div className="space-y-2 text-sm">
                                        <div className="flex justify-between">
                                            <span className="text-gray-400">Rule 2/3 Check:</span>
                                            <span className={authResult.passed_biometrics >= 2 ? "text-green-400" : "text-red-400"}>
                                                {authResult.passed_biometrics}/3 Passed
                                            </span>
                                        </div>
                                        {authResult.liveness_checks && Object.entries(authResult.liveness_checks).map(([key, res]: [string, any]) => (
                                            <div key={key} className="flex justify-between border-t border-gray-800 pt-2 mt-2">
                                                <span className="text-gray-400 capitalize">{key} Liveness:</span>
                                                <span className={res.is_live ? "text-green-400" : "text-red-400"}>
                                                    {res.is_live ? "Real" : "Spoof Detected"}
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                {authResult.success ? (
                                    <Link href="/dashboard" className="block w-full bg-green-600 hover:bg-green-500 text-white font-medium py-4 rounded-xl transition-all">
                                        Proceed to Dashboard
                                    </Link>
                                ) : (
                                    <button
                                        onClick={() => setStep(1)}
                                        className="w-full bg-gray-800 hover:bg-gray-700 text-white font-medium py-4 rounded-xl transition-all"
                                    >
                                        Try Again
                                    </button>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

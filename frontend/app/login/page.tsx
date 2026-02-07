'use client';

import { useState, useRef, useCallback } from 'react';
import Link from 'next/link';
import axios from 'axios';
import Webcam from 'react-webcam';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Camera, Upload, Check, X, Mail, ShieldCheck, ShieldAlert,
    Eye, Fingerprint, Loader2, AlertCircle, CheckCircle2,
    ArrowLeft, Settings, Sparkles
} from 'lucide-react';
import toast, { Toaster } from 'react-hot-toast';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Removed floating particles for cleaner look

export default function LoginPage() {
    const [step, setStep] = useState(0); // 0: email, 1: biometrics, 2: processing, 3: result
    const [email, setEmail] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [skipLiveness, setSkipLiveness] = useState(false); // NEW: Skip liveness for testing

    // Biometric Data
    const webcamRef = useRef<Webcam>(null);
    const [faceImage, setFaceImage] = useState<Blob | null>(null);
    const [facePreview, setFacePreview] = useState<string | null>(null);
    const [irisFile, setIrisFile] = useState<File | null>(null);
    const [fingerprintFile, setFingerprintFile] = useState<File | null>(null);

    // Biometric Status
    const [biometricStatus, setBiometricStatus] = useState({
        face: 'pending', // pending, captured, processing, success, failed
        iris: 'pending',
        fingerprint: 'pending',
    });

    // Auth Result
    const [authResult, setAuthResult] = useState<{
        success: boolean;
        message: string;
        passed_biometrics: number;
        liveness_checks?: any;
        analysis_report?: string;
    } | null>(null);

    const handleEmailSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');

        if (!email.includes('@')) {
            toast.error('Please enter a valid email address');
            return;
        }

        toast.success('Email verified. Please provide biometrics.');
        setStep(1);
    };

    const captureFace = useCallback(() => {
        if (webcamRef.current) {
            const imageSrc = webcamRef.current.getScreenshot();
            if (imageSrc) {
                setFacePreview(imageSrc);
                fetch(imageSrc)
                    .then(res => res.blob())
                    .then(blob => {
                        setFaceImage(blob);
                        setBiometricStatus(prev => ({ ...prev, face: 'captured' }));
                        toast.success('Face captured successfully');
                    });
            }
        }
    }, [webcamRef]);

    const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>, type: 'iris' | 'fingerprint') => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            if (type === 'iris') {
                setIrisFile(file);
                setBiometricStatus(prev => ({ ...prev, iris: 'captured' }));
                toast.success('Iris image uploaded');
            } else {
                setFingerprintFile(file);
                setBiometricStatus(prev => ({ ...prev, fingerprint: 'captured' }));
                toast.success('Fingerprint image uploaded');
            }
        }
    };

    const handleLogin = async () => {
        let count = 0;
        if (faceImage) count++;
        if (irisFile) count++;
        if (fingerprintFile) count++;

        if (count < 2) {
            toast.error('Please provide at least 2 biometrics (2/3 Rule)');
            return;
        }

        setLoading(true);
        setError('');
        setStep(2);

        try {
            const formData = new FormData();
            formData.append('email', email);
            formData.append('skip_liveness', String(skipLiveness)); // Send skip_liveness flag

            if (faceImage) formData.append('face_image', faceImage, 'face.jpg');
            if (irisFile) formData.append('iris_image', irisFile);
            if (fingerprintFile) formData.append('fingerprint_image', fingerprintFile);

            const response = await axios.post(`${API_BASE}/authenticate`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });

            setAuthResult(response.data);

            if (response.data.success) {
                toast.success('Authentication successful');
                setStep(3);
            } else {
                toast.error('Authentication failed');
                setStep(3);
            }
        } catch (err: any) {
            const errorMsg = err.response?.data?.message || 'Authentication failed. Please try again.';
            toast.error(errorMsg);
            setError(errorMsg);
            setStep(1);
        } finally {
            setLoading(false);
        }
    };

    const resetForm = () => {
        setStep(0);
        setEmail('');
        setFaceImage(null);
        setFacePreview(null);
        setIrisFile(null);
        setFingerprintFile(null);
        setAuthResult(null);
        setError('');
        setBiometricStatus({ face: 'pending', iris: 'pending', fingerprint: 'pending' });
    };

    // Status Icon Component
    const StatusIcon = ({ status }: { status: string }) => {
        switch (status) {
            case 'captured':
                return <CheckCircle2 className="w-5 h-5 text-green-500" />;
            case 'failed':
                return <AlertCircle className="w-5 h-5 text-red-500" />;
            case 'processing':
                return <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />;
            default:
                return <div className="w-5 h-5 border-2 border-gray-300 rounded-full" />;
        }
    };

    return (
        <div className="min-h-screen relative bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 overflow-hidden">
            <Toaster position="top-right" />

            <div className="relative z-10 min-h-screen flex items-center justify-center p-4">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="w-full max-w-5xl"
                >
                    {/* Header */}
                    <div className="text-center mb-8">
                        <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            transition={{ type: 'spring', stiffness: 200 }}
                            className="mb-4"
                        >
                            <h1 className="text-5xl font-bold text-white">
                                Biometric Login
                            </h1>
                        </motion.div>
                        <p className="text-gray-300 text-lg">Secure 2/3 Multi-Factor Authentication</p>
                    </div>

                    {/* Main Card */}
                    <motion.div
                        layout
                        className="bg-white/10 backdrop-blur-xl rounded-3xl border border-white/20 shadow-2xl p-8"
                    >
                        <AnimatePresence mode="wait">
                            {/* Step 0: Email Input */}
                            {step === 0 && (
                                <motion.div
                                    key="email-step"
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: 20 }}
                                    className="space-y-6"
                                >
                                    <div className="text-center">
                                        <h2 className="text-2xl font-bold text-white mb-2">Enter Your Email</h2>
                                        <p className="text-gray-400">Start your secure authentication</p>
                                    </div>

                                    <form onSubmit={handleEmailSubmit} className="space-y-4">
                                        <div>
                                            <input
                                                type="email"
                                                placeholder="your.email@example.com"
                                                value={email}
                                                onChange={(e) => setEmail(e.target.value)}
                                                className="w-full px-6 py-4 bg-white/10 border border-white/20 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                                                required
                                            />
                                        </div>

                                        <motion.button
                                            whileHover={{ scale: 1.02 }}
                                            whileTap={{ scale: 0.98 }}
                                            type="submit"
                                            className="w-full bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white font-semibold py-4 rounded-xl transition-all shadow-lg shadow-blue-500/50"
                                        >
                                            Continue
                                        </motion.button>
                                    </form>

                                    <div className="text-center">
                                        <p className="text-gray-400 text-sm">
                                            Don't have an account?{' '}
                                            <Link href="/register" className="text-blue-400 hover:text-blue-300 transition-colors">
                                                Register here
                                            </Link>
                                        </p>
                                    </div>
                                </motion.div>
                            )}

                            {/*Step 1: Biometrics Collection */}
                            {step === 1 && (
                                <motion.div
                                    key="biometric-step"
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: 20 }}
                                    className="space-y-6"
                                >
                                    {/* Header with Email */}
                                    <div className="flex items-center justify-between mb-6">
                                        <button
                                            onClick={() => setStep(0)}
                                            className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
                                        >
                                            <ArrowLeft className="w-5 h-5" />
                                            Back
                                        </button>
                                        <div className="flex items-center gap-2 text-white">
                                            <Mail className="w-5 h-5 text-blue-400" />
                                            <span className="font-medium">{email}</span>
                                        </div>
                                    </div>

                                    {/* Skip Liveness Toggle */}
                                    <motion.div
                                        initial={{ opacity: 0, y: -10 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        className="flex items-center justify-between p-4 bg-slate-800/50 border border-slate-700 rounded-xl"
                                    >
                                        <div className="flex items-center gap-3">
                                            <div>
                                                <p className="text-white font-medium">Testing Mode</p>
                                                <p className="text-gray-400 text-sm">Skip liveness detection for dataset images</p>
                                            </div>
                                        </div>
                                        <label className="relative inline-flex items-center cursor-pointer">
                                            <input
                                                type="checkbox"
                                                checked={skipLiveness}
                                                onChange={(e) => {
                                                    setSkipLiveness(e.target.checked);
                                                    if (e.target.checked) {
                                                        toast.success('Testing mode enabled');
                                                    } else {
                                                        toast.success('Testing mode disabled');
                                                    }
                                                }}
                                                className="sr-only peer"
                                            />
                                            <div className="w-14 h-7 bg-gray-700 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-amber-500/50 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-6 after:w-6 after:transition-all peer-checked:bg-amber-500"></div>
                                        </label>
                                    </motion.div>

                                    {/* Biometric Collection Grid */}
                                    <div className="grid md:grid-cols-3 gap-6">
                                        {/* Face Capture */}
                                        <motion.div
                                            whileHover={{ scale: 1.02 }}
                                            className="bg-white/5 border border-white/10 rounded-2xl p-6 space-y-4"
                                        >
                                            <div className="flex items-center justify-between">
                                                <div>
                                                    <h3 className="text-white font-semibold">Face</h3>
                                                </div>
                                                <StatusIcon status={biometricStatus.face} />
                                            </div>

                                            {facePreview ? (
                                                <div className="relative">
                                                    <img src={facePreview} alt="Face" className="w-full rounded-xl" />
                                                    <button
                                                        onClick={() => {
                                                            setFaceImage(null);
                                                            setFacePreview(null);
                                                            setBiometricStatus(prev => ({ ...prev, face: 'pending' }));
                                                        }}
                                                        className="absolute top-2 right-2 p-2 bg-red-500 rounded-full hover:bg-red-600 transition-colors"
                                                    >
                                                        <X className="w-4 h-4 text-white" />
                                                    </button>
                                                </div>
                                            ) : (
                                                <div className="space-y-3">
                                                    <div className="aspect-square bg-black/30 rounded-xl overflow-hidden">
                                                        <Webcam
                                                            ref={webcamRef}
                                                            screenshotFormat="image/jpeg"
                                                            className="w-full h-full object-cover"
                                                        />
                                                    </div>
                                                    <motion.button
                                                        whileHover={{ scale: 1.05 }}
                                                        whileTap={{ scale: 0.95 }}
                                                        onClick={captureFace}
                                                        className="w-full bg-blue-500 hover:bg-blue-600 text-white font-medium py-2 rounded-lg transition-colors flex items-center justify-center gap-2"
                                                    >
                                                        <Camera className="w-4 h-4" />
                                                        Capture
                                                    </motion.button>
                                                </div>
                                            )}
                                        </motion.div>

                                        {/* Iris Upload */}
                                        <motion.div
                                            whileHover={{ scale: 1.02 }}
                                            className="bg-white/5 border border-white/10 rounded-2xl p-6 space-y-4"
                                        >
                                            <div className="flex items-center justify-between">
                                                <div>
                                                    <h3 className="text-white font-semibold">Iris</h3>
                                                </div>
                                                <StatusIcon status={biometricStatus.iris} />
                                            </div>

                                            <div className="aspect-square bg-black/30 rounded-xl flex items-center justify-center">
                                                {irisFile ? (
                                                    <div className="relative w-full h-full">
                                                        <img
                                                            src={URL.createObjectURL(irisFile)}
                                                            alt="Iris"
                                                            className="w-full h-full object-cover rounded-xl"
                                                        />
                                                        <button
                                                            onClick={() => {
                                                                setIrisFile(null);
                                                                setBiometricStatus(prev => ({ ...prev, iris: 'pending' }));
                                                            }}
                                                            className="absolute top-2 right-2 p-2 bg-red-500 rounded-full hover:bg-red-600 transition-colors"
                                                        >
                                                            <X className="w-4 h-4 text-white" />
                                                        </button>
                                                    </div>
                                                ) : (
                                                    <Eye className="w-16 h-16 text-gray-600" />
                                                )}
                                            </div>

                                            <label className="block">
                                                <input
                                                    type="file"
                                                    accept="image/*"
                                                    onChange={(e) => handleFileUpload(e, 'iris')}
                                                    className="hidden"
                                                />
                                                <motion.div
                                                    whileHover={{ scale: 1.05 }}
                                                    whileTap={{ scale: 0.95 }}
                                                    className="w-full bg-cyan-500 hover:bg-cyan-600 text-white font-medium py-2 rounded-lg transition-colors flex items-center justify-center gap-2 cursor-pointer"
                                                >
                                                    <Upload className="w-4 h-4" />
                                                    Upload
                                                </motion.div>
                                            </label>
                                        </motion.div>

                                        {/* Fingerprint Upload */}
                                        <motion.div
                                            whileHover={{ scale: 1.02 }}
                                            className="bg-white/5 border border-white/10 rounded-2xl p-6 space-y-4"
                                        >
                                            <div className="flex items-center justify-between">
                                                <div>
                                                    <h3 className="text-white font-semibold">Fingerprint</h3>
                                                </div>
                                                <StatusIcon status={biometricStatus.fingerprint} />
                                            </div>

                                            <div className="aspect-square bg-black/30 rounded-xl flex items-center justify-center">
                                                {fingerprintFile ? (
                                                    <div className="relative w-full h-full">
                                                        <img
                                                            src={URL.createObjectURL(fingerprintFile)}
                                                            alt="Fingerprint"
                                                            className="w-full h-full object-cover rounded-xl"
                                                        />
                                                        <button
                                                            onClick={() => {
                                                                setFingerprintFile(null);
                                                                setBiometricStatus(prev => ({ ...prev, fingerprint: 'pending' }));
                                                            }}
                                                            className="absolute top-2 right-2 p-2 bg-red-500 rounded-full hover:bg-red-600 transition-colors"
                                                        >
                                                            <X className="w-4 h-4 text-white" />
                                                        </button>
                                                    </div>
                                                ) : (
                                                    <Fingerprint className="w-16 h-16 text-gray-600" />
                                                )}
                                            </div>

                                            <label className="block">
                                                <input
                                                    type="file"
                                                    accept="image/*"
                                                    onChange={(e) => handleFileUpload(e, 'fingerprint')}
                                                    className="hidden"
                                                />
                                                <motion.div
                                                    whileHover={{ scale: 1.05 }}
                                                    whileTap={{ scale: 0.95 }}
                                                    className="w-full bg-purple-500 hover:bg-purple-600 text-white font-medium py-2 rounded-lg transition-colors flex items-center justify-center gap-2 cursor-pointer"
                                                >
                                                    <Upload className="w-4 h-4" />
                                                    Upload
                                                </motion.div>
                                            </label>
                                        </motion.div>
                                    </div>

                                    {/* Submit Button */}
                                    <motion.button
                                        whileHover={{ scale: 1.02 }}
                                        whileTap={{ scale: 0.98 }}
                                        onClick={handleLogin}
                                        disabled={loading}
                                        className="w-full bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white font-bold py-4 rounded-xl transition-all shadow-lg shadow-green-500/50 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                                    >
                                        {loading ? (
                                            <>
                                                <Loader2 className="w-5 h-5 animate-spin" />
                                                Authenticating...
                                            </>
                                        ) : (
                                            <>
                                                <ShieldCheck className="w-5 h-5" />
                                                Authenticate
                                            </>
                                        )}
                                    </motion.button>

                                    <p className="text-center text-gray-400 text-sm">
                                        Provide at least 2 biometrics to login (2/3 Rule)
                                    </p>
                                </motion.div>
                            )}

                            {/* Step 2: Processing */}
                            {step === 2 && (
                                <motion.div
                                    key="processing-step"
                                    initial={{ opacity: 0, scale: 0.9 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    exit={{ opacity: 0, scale: 1.1 }}
                                    className="text-center py-12 space-y-6"
                                >
                                    <motion.div
                                        animate={{ rotate: 360 }}
                                        transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                                        className="inline-block"
                                    >
                                        <ShieldCheck className="w-20 h-20 text-blue-400" />
                                    </motion.div>
                                    <div>
                                        <h2 className="text-3xl font-bold text-white mb-2">Authenticating...</h2>
                                        <p className="text-gray-400">Analyzing your biometric data</p>
                                    </div>
                                    <div className="flex justify-center gap-2">
                                        <motion.div
                                            animate={{ scale: [1, 1.2, 1] }}
                                            transition={{ duration: 1, repeat: Infinity, delay: 0 }}
                                            className="w-3 h-3 bg-blue-500 rounded-full"
                                        />
                                        <motion.div
                                            animate={{ scale: [1, 1.2, 1] }}
                                            transition={{ duration: 1, repeat: Infinity, delay: 0.2 }}
                                            className="w-3 h-3 bg-cyan-500 rounded-full"
                                        />
                                        <motion.div
                                            animate={{ scale: [1, 1.2, 1] }}
                                            transition={{ duration: 1, repeat: Infinity, delay: 0.4 }}
                                            className="w-3 h-3 bg-purple-500 rounded-full"
                                        />
                                    </div>
                                </motion.div>
                            )}

                            {/* Step 3: Result */}
                            {step === 3 && authResult && (
                                <motion.div
                                    key="result-step"
                                    initial={{ opacity: 0, scale: 0.9 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    className="space-y-6"
                                >
                                    {/* Result Icon */}
                                    <div className="text-center">
                                        {authResult.success ? (
                                            <motion.div
                                                initial={{ scale: 0 }}
                                                animate={{ scale: 1 }}
                                                transition={{ type: 'spring', stiffness: 200 }}
                                            >
                                                <ShieldCheck className="w-24 h-24 mx-auto text-green-500 mb-4" />
                                                <h2 className="text-3xl font-bold text-white mb-2">Access Granted!</h2>
                                                <p className="text-gray-400">Authentication successful</p>
                                            </motion.div>
                                        ) : (
                                            <motion.div
                                                initial={{ scale: 0 }}
                                                animate={{ scale: 1 }}
                                                transition={{ type: 'spring', stiffness: 200 }}
                                            >
                                                <ShieldAlert className="w-24 h-24 mx-auto text-red-500 mb-4" />
                                                <h2 className="text-3xl font-bold text-white mb-2">Access Denied</h2>
                                                <p className="text-gray-400">{authResult.message}</p>
                                            </motion.div>
                                        )}
                                    </div>

                                    {/* Stats */}
                                    <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
                                        <h3 className="text-white font-semibold mb-4 text-center">Authentication Report</h3>
                                        <div className="grid grid-cols-3 gap-4 text-center">
                                            <div>
                                                <p className="text-2xl font-bold text-blue-400">{authResult.passed_biometrics}</p>
                                                <p className="text-gray-400 text-sm">Passed</p>
                                            </div>
                                            <div>
                                                <p className="text-2xl font-bold text-gray-400">/</p>
                                                <p className="text-gray-400 text-sm">of</p>
                                            </div>
                                            <div>
                                                <p className="text-2xl font-bold text-purple-400">3</p>
                                                <p className="text-gray-400 text-sm">Total</p>
                                            </div>
                                        </div>

                                        {/* Liveness Results */}
                                        {authResult.liveness_checks && (
                                            <div className="mt-6 space-y-2">
                                                <h4 className="text-white font-medium text-sm mb-3">Liveness Checks:</h4>
                                                {Object.entries(authResult.liveness_checks).map(([key, value]: [string, any]) => (
                                                    <div key={key} className="flex items-center justify-between text-sm">
                                                        <span className="text-gray-400 capitalize">{key}:</span>
                                                        <span className={value?.is_live ? 'text-green-400' : 'text-red-400'}>
                                                            {value?.is_live || value?.skipped ? 'Real' : 'Spoof Detected'}
                                                        </span>
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </div>

                                    {/* Action Buttons */}
                                    <div className="flex gap-4">
                                        <motion.button
                                            whileHover={{ scale: 1.05 }}
                                            whileTap={{ scale: 0.95 }}
                                            onClick={resetForm}
                                            className="flex-1 bg-blue-500 hover:bg-blue-600 text-white font-semibold py-3 rounded-xl transition-colors"
                                        >
                                            Try Again
                                        </motion.button>
                                        {authResult.success && (
                                            <motion.button
                                                whileHover={{ scale: 1.05 }}
                                                whileTap={{ scale: 0.95 }}
                                                onClick={() => window.location.href = '/'}
                                                className="flex-1 bg-green-500 hover:bg-green-600 text-white font-semibold py-3 rounded-xl transition-colors"
                                            >
                                                Continue to Dashboard
                                            </motion.button>
                                        )}
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </motion.div>

                    {/* Footer */}
                    <div className="text-center mt-8">
                        <p className="text-gray-400 text-sm">
                            Powered by InsightFace, Gabor Wavelets & SIFT
                        </p>
                    </div>
                </motion.div>
            </div>
        </div>
    );
}

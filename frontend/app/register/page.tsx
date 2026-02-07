'use client';

import { useState, useRef, useCallback } from 'react';
import Link from 'next/link';
import axios from 'axios';
import Webcam from 'react-webcam';
import { Camera, Upload, Check, X, User, Mail, Calendar, Trash2, ArrowRight, ArrowLeft, CheckCircle2 } from 'lucide-react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function RegisterPage() {
    const [step, setStep] = useState(0);
    const [formData, setFormData] = useState({
        name: '',
        age: '',
        email: ''
    });

    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const webcamRef = useRef<Webcam>(null);
    const [capturedImages, setCapturedImages] = useState<string[]>([]);
    const [irisFiles, setIrisFiles] = useState<File[]>([]);
    const [fpFiles, setFpFiles] = useState<File[]>([]);

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleInfoSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');

        if (!formData.email || !formData.name || !formData.age) {
            setError('Please fill in all fields');
            return;
        }

        setLoading(true);

        try {
            const response = await axios.post(`${API_BASE}/register/user`, {
                name: formData.name,
                age: parseInt(formData.age),
                email: formData.email
            }, {
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
            });

            if (response.data.success) {
                setStep(1);
            } else {
                setError(response.data.message || 'Registration failed');
            }
        } catch (err: any) {
            setError(err.response?.data?.message || 'Error connecting to server');
        } finally {
            setLoading(false);
        }
    };

    const captureFace = useCallback(() => {
        if (webcamRef.current) {
            const imageSrc = webcamRef.current.getScreenshot();
            if (imageSrc) {
                setCapturedImages(prev => [...prev, imageSrc]);
            }
        }
    }, [webcamRef]);

    const deleteCapturedImage = (index: number) => {
        setCapturedImages(prev => prev.filter((_, i) => i !== index));
    };

    const handleFaceSubmit = async () => {
        if (capturedImages.length < 5) {
            setError('Please capture at least 5 photos');
            return;
        }

        setLoading(true);
        setError('');

        try {
            const formDataUpload = new FormData();
            formDataUpload.append('email', formData.email);

            for (let i = 0; i < capturedImages.length; i++) {
                const fetchRes = await fetch(capturedImages[i]);
                const blob = await fetchRes.blob();
                formDataUpload.append('images', blob, `face_${i}.jpg`);
            }

            const response = await axios.post(`${API_BASE}/register/face`, formDataUpload, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });

            if (response.data.success) {
                setStep(2);
                setCapturedImages([]);
            } else {
                setError(response.data.message || 'Failed to upload face images');
            }
        } catch (err: any) {
            setError(err.response?.data?.message || 'Error uploading images');
        } finally {
            setLoading(false);
        }
    };

    const handleIrisFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
            setIrisFiles(Array.from(e.target.files));
        }
    };

    const deleteIrisFile = (index: number) => {
        setIrisFiles(prev => prev.filter((_, i) => i !== index));
    };

    const handleIrisSubmit = async () => {
        if (irisFiles.length < 3) {
            setError('Please upload at least 3 iris images');
            return;
        }

        setLoading(true);
        setError('');

        try {
            const formDataUpload = new FormData();
            formDataUpload.append('email', formData.email);
            irisFiles.forEach(file => {
                formDataUpload.append('images', file);
            });

            const response = await axios.post(`${API_BASE}/register/iris`, formDataUpload, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });

            if (response.data.success) {
                setStep(3);
                setIrisFiles([]);
            } else {
                setError(response.data.message || 'Failed to upload iris images');
            }
        } catch (err: any) {
            setError(err.response?.data?.message || 'Error uploading images');
        } finally {
            setLoading(false);
        }
    };

    const handleFpFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
            setFpFiles(Array.from(e.target.files));
        }
    };

    const deleteFpFile = (index: number) => {
        setFpFiles(prev => prev.filter((_, i) => i !== index));
    };

    const handleFpSubmit = async () => {
        if (fpFiles.length < 3) {
            setError('Please upload at least 3 fingerprint images');
            return;
        }

        setLoading(true);
        setError('');

        try {
            const formDataUpload = new FormData();
            formDataUpload.append('email', formData.email);
            fpFiles.forEach(file => {
                formDataUpload.append('images', file);
            });

            const response = await axios.post(`${API_BASE}/register/fingerprint`, formDataUpload, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });

            if (response.data.success) {
                setStep(4);
                setFpFiles([]);
            } else {
                setError(response.data.message || 'Failed to upload fingerprint images');
            }
        } catch (err: any) {
            setError(err.response?.data?.message || 'Error uploading images');
        } finally {
            setLoading(false);
        }
    };

    const steps = [
        { num: 1, label: 'Info', icon: User },
        { num: 2, label: 'Face', icon: Camera },
        { num: 3, label: 'Iris', icon: Camera },
        { num: 4, label: 'Fingerprint', icon: Camera }
    ];

    return (
        <div className="min-h-screen bg-slate-900 flex items-center justify-center p-4">
            <div className="w-full max-w-4xl">
                <Link href="/" className="inline-flex items-center text-slate-400 hover:text-white mb-6 transition-colors">
                    <ArrowLeft className="mr-2" size={20} />
                    Back to Home
                </Link>

                <div className="bg-slate-800 rounded-2xl shadow-xl border border-slate-700 p-8 md:p-12">
                    <div className="text-center mb-8">
                        <h1 className="text-3xl md:text-4xl font-bold text-white mb-2">
                            Create Account
                        </h1>
                        <p className="text-slate-400">Register with Multi-Factor Biometrics</p>
                    </div>

                    {/* Progress Steps */}
                    <div className="flex justify-between items-center mb-12 relative">
                        <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-slate-700 -translate-y-1/2">
                            <div
                                className="h-full bg-blue-500 transition-all duration-500"
                                style={{ width: `${(step / 4) * 100}%` }}
                            ></div>
                        </div>
                        {steps.map((s, idx) => {
                            const Icon = s.icon;
                            const isActive = step >= idx;
                            const isCurrent = step === idx;
                            return (
                                <div key={idx} className="relative flex flex-col items-center z-10">
                                    <div className={`
                                        w-12 h-12 rounded-full flex items-center justify-center mb-2 transition-all
                                        ${isActive
                                            ? 'bg-blue-500 shadow-lg'
                                            : 'bg-slate-700'
                                        }
                                        ${isCurrent ? 'ring-4 ring-blue-500/30' : ''}
                                    `}>
                                        {isActive && idx < step ? (
                                            <CheckCircle2 className="text-white" size={20} />
                                        ) : (
                                            <Icon className={isActive ? 'text-white' : 'text-slate-400'} size={20} />
                                        )}
                                    </div>
                                    <span className={`text-sm ${isActive ? 'text-white' : 'text-slate-500'}`}>
                                        {s.label}
                                    </span>
                                </div>
                            );
                        })}
                    </div>

                    {/* Error Alert */}
                    {error && (
                        <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-lg flex items-start">
                            <X className="text-red-400 mr-3 flex-shrink-0 mt-0.5" size={20} />
                            <p className="text-red-200">{error}</p>
                        </div>
                    )}

                    {/* Step 0: Info */}
                    {step === 0 && (
                        <form onSubmit={handleInfoSubmit} className="space-y-5">
                            <div>
                                <label className="block text-slate-300 mb-2 text-sm font-medium">Full Name</label>
                                <div className="relative">
                                    <User className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                                    <input
                                        type="text"
                                        name="name"
                                        value={formData.name}
                                        onChange={handleInputChange}
                                        className="w-full bg-slate-700 border border-slate-600 rounded-lg pl-10 pr-4 py-3 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                        placeholder="Enter your full name"
                                    />
                                </div>
                            </div>

                            <div>
                                <label className="block text-slate-300 mb-2 text-sm font-medium">Age</label>
                                <div className="relative">
                                    <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                                    <input
                                        type="number"
                                        name="age"
                                        value={formData.age}
                                        onChange={handleInputChange}
                                        className="w-full bg-slate-700 border border-slate-600 rounded-lg pl-10 pr-4 py-3 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                        placeholder="Enter your age"
                                    />
                                </div>
                            </div>

                            <div>
                                <label className="block text-slate-300 mb-2 text-sm font-medium">Email Address</label>
                                <div className="relative">
                                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                                    <input
                                        type="email"
                                        name="email"
                                        value={formData.email}
                                        onChange={handleInputChange}
                                        className="w-full bg-slate-700 border border-slate-600 rounded-lg pl-10 pr-4 py-3 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                        placeholder="Enter your email"
                                    />
                                </div>
                            </div>

                            <button
                                type="submit"
                                disabled={loading}
                                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                            >
                                {loading ? 'Processing...' : 'Continue to Face Scan'}
                            </button>
                        </form>
                    )}

                    {/* Step 1: Face */}
                    {step === 1 && (
                        <div className="space-y-6">
                            <div className="text-center mb-4">
                                <h2 className="text-2xl font-bold text-white mb-1">Face Registration</h2>
                                <p className="text-slate-400">Position your face in the camera and capture 5 photos</p>
                            </div>

                            <div className="rounded-xl overflow-hidden bg-black border border-slate-700">
                                <Webcam
                                    ref={webcamRef}
                                    screenshotFormat="image/jpeg"
                                    className="w-full h-auto"
                                    videoConstraints={{
                                        width: 1280,
                                        height: 720,
                                        facingMode: 'user'
                                    }}
                                />
                            </div>

                            {capturedImages.length > 0 && (
                                <div className="grid grid-cols-5 gap-3">
                                    {capturedImages.map((img, idx) => (
                                        <div key={idx} className="relative group">
                                            <img
                                                src={img}
                                                alt={`Face ${idx + 1}`}
                                                className="w-full aspect-square object-cover rounded-lg border-2 border-blue-500"
                                            />
                                            <button
                                                onClick={() => deleteCapturedImage(idx)}
                                                className="absolute -top-2 -right-2 bg-red-600 hover:bg-red-700 text-white rounded-full p-1.5 shadow-lg opacity-0 group-hover:opacity-100 transition-opacity"
                                            >
                                                <Trash2 size={14} />
                                            </button>
                                        </div>
                                    ))}
                                </div>
                            )}

                            <div className="flex gap-3">
                                <button
                                    onClick={captureFace}
                                    className="flex-1 bg-slate-700 hover:bg-slate-600 border border-slate-600 text-white font-medium py-3 rounded-lg transition-colors flex items-center justify-center"
                                >
                                    <Camera className="mr-2" size={18} />
                                    Capture ({capturedImages.length}/5)
                                </button>
                                <button
                                    onClick={() => setCapturedImages([])}
                                    disabled={capturedImages.length === 0}
                                    className="px-6 bg-red-600/20 hover:bg-red-600/30 border border-red-600/40 text-red-200 font-medium py-3 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    Retake All
                                </button>
                            </div>

                            <button
                                onClick={handleFaceSubmit}
                                disabled={loading || capturedImages.length < 5}
                                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                            >
                                {loading ? 'Uploading...' : 'Continue to Iris Scan'}
                            </button>
                        </div>
                    )}

                    {/* Step 2: Iris */}
                    {step === 2 && (
                        <div className="space-y-6">
                            <div className="text-center mb-4">
                                <h2 className="text-2xl font-bold text-white mb-1">Iris Registration</h2>
                                <p className="text-slate-400">Upload at least 3 clear images</p>
                            </div>

                            <div className="border-2 border-dashed border-slate-600 rounded-xl p-8 text-center bg-slate-700/30 hover:border-blue-500 transition-colors">
                                <input
                                    type="file"
                                    multiple
                                    accept="image/*"
                                    onChange={handleIrisFileChange}
                                    className="hidden"
                                    id="iris-upload"
                                />
                                <label htmlFor="iris-upload" className="cursor-pointer">
                                    <Upload className="mx-auto mb-3 text-slate-400" size={40} />
                                    <p className="text-white font-medium mb-1">Click to Upload</p>
                                    <p className="text-slate-400 text-sm">JPG, PNG supported</p>
                                </label>
                            </div>

                            {irisFiles.length > 0 && (
                                <div>
                                    <p className="text-slate-300 mb-3 text-sm">Selected: {irisFiles.length} files</p>
                                    <div className="grid grid-cols-3 gap-3">
                                        {irisFiles.map((file, idx) => (
                                            <div key={idx} className="relative group">
                                                <img
                                                    src={URL.createObjectURL(file)}
                                                    alt={`Iris ${idx + 1}`}
                                                    className="w-full aspect-square object-cover rounded-lg border-2 border-blue-500"
                                                />
                                                <button
                                                    onClick={() => deleteIrisFile(idx)}
                                                    className="absolute -top-2 -right-2 bg-red-600 hover:bg-red-700 text-white rounded-full p-1.5 shadow-lg opacity-0 group-hover:opacity-100 transition-opacity"
                                                >
                                                    <Trash2 size={14} />
                                                </button>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            <button
                                onClick={handleIrisSubmit}
                                disabled={loading || irisFiles.length < 3}
                                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                            >
                                {loading ? 'Uploading...' : 'Continue to Fingerprint'}
                            </button>
                        </div>
                    )}

                    {/* Step 3: Fingerprint */}
                    {step === 3 && (
                        <div className="space-y-6">
                            <div className="text-center mb-4">
                                <h2 className="text-2xl font-bold text-white mb-1">Fingerprint Registration</h2>
                                <p className="text-slate-400">Upload at least 3 clear images</p>
                            </div>

                            <div className="border-2 border-dashed border-slate-600 rounded-xl p-8 text-center bg-slate-700/30 hover:border-blue-500 transition-colors">
                                <input
                                    type="file"
                                    multiple
                                    accept="image/*"
                                    onChange={handleFpFileChange}
                                    className="hidden"
                                    id="fp-upload"
                                />
                                <label htmlFor="fp-upload" className="cursor-pointer">
                                    <Upload className="mx-auto mb-3 text-slate-400" size={40} />
                                    <p className="text-white font-medium mb-1">Click to Upload</p>
                                    <p className="text-slate-400 text-sm">JPG, PNG supported</p>
                                </label>
                            </div>

                            {fpFiles.length > 0 && (
                                <div>
                                    <p className="text-slate-300 mb-3 text-sm">Selected: {fpFiles.length} files</p>
                                    <div className="grid grid-cols-3 gap-3">
                                        {fpFiles.map((file, idx) => (
                                            <div key={idx} className="relative group">
                                                <img
                                                    src={URL.createObjectURL(file)}
                                                    alt={`Fingerprint ${idx + 1}`}
                                                    className="w-full aspect-square object-cover rounded-lg border-2 border-blue-500"
                                                />
                                                <button
                                                    onClick={() => deleteFpFile(idx)}
                                                    className="absolute -top-2 -right-2 bg-red-600 hover:bg-red-700 text-white rounded-full p-1.5 shadow-lg opacity-0 group-hover:opacity-100 transition-opacity"
                                                >
                                                    <Trash2 size={14} />
                                                </button>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            <button
                                onClick={handleFpSubmit}
                                disabled={loading || fpFiles.length < 3}
                                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                            >
                                {loading ? 'Uploading...' : 'Complete Registration'}
                            </button>
                        </div>
                    )}

                    {/* Step 4: Success */}
                    {step === 4 && (
                        <div className="text-center py-12">
                            <div className="mb-6 inline-block">
                                <div className="w-20 h-20 bg-green-500 rounded-full flex items-center justify-center">
                                    <CheckCircle2 className="text-white" size={40} />
                                </div>
                            </div>
                            <h2 className="text-3xl font-bold text-white mb-3">Registration Complete</h2>
                            <p className="text-slate-400 text-lg mb-8">
                                Your account has been created successfully.<br />
                                You can now log in using your email and biometrics.
                            </p>
                            <Link
                                href="/login"
                                className="inline-flex items-center bg-blue-600 hover:bg-blue-700 text-white font-medium px-8 py-3 rounded-lg transition-colors"
                            >
                                Go to Login
                                <ArrowRight className="ml-2" size={18} />
                            </Link>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

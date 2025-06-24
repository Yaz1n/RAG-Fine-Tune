import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

const UploadPage = () => {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    setFiles(selectedFiles);
  };

  const handleUpload = () => {
    if (files.length === 0) {
      alert("Please select files to upload.");
      return;
    }

    setLoading(true);

    // Simulate reading time before redirecting
    setTimeout(() => {
      // In a real app, you'd store file data or send to backend here
      navigate("/question"); // Route to AskPage
    }, 2000);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 px-4">
      <h2 className="text-3xl font-semibold text-indigo-700 mb-6">Upload QA Dataset</h2>
      <div className="bg-white shadow-lg rounded-lg p-6 w-full max-w-md text-center">
        <input
          type="file"
          multiple
          accept=".json,.csv,.doc,.pdf"
          onChange={handleFileChange}
          className="block w-full text-sm text-gray-700 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-600 file:text-white hover:file:bg-indigo-700"
        />
        {files.length > 0 && (
          <div className="mt-4 text-green-600 text-left">
            <p className="font-semibold">Selected Files:</p>
            <ul className="list-disc list-inside">
              {files.map((file, index) => (
                <li key={index}>{file.name}</li>
              ))}
            </ul>
          </div>
        )}

        <button
          onClick={handleUpload}
          disabled={loading}
          className="mt-6 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-2 px-6 rounded-full transition duration-200 disabled:opacity-50"
        >
          {loading ? "Processing..." : "Upload"}
        </button>

        {loading && (
          <div className="mt-4">
            <div className="w-6 h-6 border-4 border-indigo-500 border-dashed rounded-full animate-spin mx-auto"></div>
            <p className="text-sm text-gray-600 mt-2">Reading your dataset...</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default UploadPage;

import { useState } from "react";
import PredictionForm from "./PredictionForm";
import ResultsView from "./ResultsView";

export interface PredictionResult {
  predicted_band_gap: number;
  uncertainty: number;
  classification: string;
  similar_materials: {
    material_id: string;
    formula: string;
    band_gap: number;
  }[];
}

export default function App() {
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async (payload: {
    cif?: string;
    material_id?: string;
  }) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }

      setResult(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto px-4 py-8">
      <header className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">
          Materials Band Gap Predictor
        </h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          Predict the electronic band gap of a crystal structure using a graph
          neural network trained on Materials Project data.
        </p>
      </header>

      <PredictionForm onPredict={handlePredict} loading={loading} />

      {error && (
        <div className="mt-6 p-4 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-300">
          {error}
        </div>
      )}

      {result && <ResultsView result={result} />}
    </div>
  );
}

import React, { useState } from 'react';
import PredictionForm from './components/PredictionForm';
import PriceDistributionChart from './components/PriceDistributionChart';
import ActualVsPredictedChart from './components/ActualVsPredictedChart';
import AveragePriceByYearChart from './components/AveragePriceByYearChart';
import './App.css';

const App = () => {
  const [view, setView] = useState("prediction");
  const [selectedSuburb, setSelectedSuburb] = useState(null);
  const [selectedVisualization, setSelectedVisualization] = useState("");

  return (
    <div className="container">
      <header className="header">
        <h1 className="group-name">Fantastic Realtors - Housing Prediction System</h1>
        <div className="nav-buttons">
          <button onClick={() => setView("prediction")}>Prediction</button>
          <button onClick={() => setView("visualization")}>Visualization</button>
        </div>
      </header>

      <main className="content centered-content">
        {view === "prediction" && <PredictionForm onSuburbSelect={setSelectedSuburb} />}
        {view === "visualization" && (
          <div style={{ textAlign: 'center' }}>
            <h2 style={{ color: 'var(--primary-color)', marginBottom: '20px' }}>Select a Visualization</h2>
            <select
              onChange={(e) => setSelectedVisualization(e.target.value)}
              value={selectedVisualization}
              style={{ padding: '0.5rem', borderRadius: '5px', marginBottom: '20px' }}
            >
              <option value="">Choose a Visualization</option>
              <option value="priceDistribution">Price Distribution by Suburb</option>
              <option value="actualVsPredicted">Actual vs. Predicted Price</option>
              <option value="averagePriceByYear">Average Price by Year Built</option>
            </select>

            {selectedVisualization === "priceDistribution" && <PriceDistributionChart suburb={selectedSuburb} />}
            {selectedVisualization === "actualVsPredicted" && <ActualVsPredictedChart suburb={selectedSuburb} />}
            {selectedVisualization === "averagePriceByYear" && <AveragePriceByYearChart suburb={selectedSuburb} />}
          </div>
        )}
      </main>


    </div>
  );
};

export default App;
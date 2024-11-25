import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import axios from 'axios';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    LineElement,
    PointElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, LineElement, PointElement, Title, Tooltip, Legend);

const ActualVsPredictedChart = () => {
    const [suburbs, setSuburbs] = useState([]);              // List of suburbs
    const [selectedSuburb, setSelectedSuburb] = useState(''); // Selected suburb
    const [chartData, setChartData] = useState(null);
    const [error, setError] = useState(null);

    // Fetch all available suburbs on component mount
    useEffect(() => {
        const fetchSuburbs = async () => {
            try {
                const response = await axios.get('http://127.0.0.1:8000/suburbs');
                setSuburbs(response.data.suburbs);
            } catch (error) {
                console.error("Error fetching suburbs:", error);
                setError("Failed to load suburbs. Please check the server.");
            }
        };
        fetchSuburbs();
    }, []);

    // Fetch actual vs predicted prices for the selected suburb
    useEffect(() => {
        if (selectedSuburb) {
            const fetchData = async () => {
                try {
                    const response = await axios.get(`http://127.0.0.1:8000/actual-vs-predicted/${selectedSuburb}`);
                    const { actual, predicted } = response.data;

                    const formattedData = {
                        labels: actual.map((_, idx) => `Data Point ${idx + 1}`),
                        datasets: [
                            {
                                label: 'Actual Price',
                                data: actual,
                                borderColor: 'rgba(255,99,132,1)',
                                backgroundColor: 'rgba(255,99,132,0.2)',
                                fill: false,
                                tension: 0.1,
                            },
                            {
                                label: 'Predicted Price',
                                data: predicted,
                                borderColor: 'rgba(54, 162, 235, 1)',
                                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                                fill: false,
                                tension: 0.1,
                            },
                        ],
                    };

                    setChartData(formattedData);
                } catch (error) {
                    console.error("Error fetching actual vs predicted data:", error);
                    setError("Failed to load actual vs predicted data.");
                }
            };
            fetchData();
        }
    }, [selectedSuburb]);

    return (
        <div style={{ width: '600px', height: '400px', margin: 'auto' }}>
            <h3>Actual vs Predicted Price</h3>

            <label>Select Suburb:</label>
            <select onChange={(e) => setSelectedSuburb(e.target.value)} value={selectedSuburb || ''}>
                <option value="">Select a Suburb</option>
                {suburbs.map((suburb) => (
                    <option key={suburb} value={suburb}>
                        {suburb}
                    </option>
                ))}
            </select>

            {error && <p style={{ color: 'red' }}>{error}</p>}
            {chartData && (
                <Line
                    data={chartData}
                    options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: { display: true, text: 'Price' },
                            },
                        },
                    }}
                />
            )}
        </div>
    );
};

export default ActualVsPredictedChart;

import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import axios from 'axios';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    LineElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, LineElement, Title, Tooltip, Legend);

const AveragePriceByYearChart = () => {
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

    // Fetch average price by year for the selected suburb
    useEffect(() => {
        if (selectedSuburb) {
            const fetchAveragePriceByYear = async () => {
                try {
                    const response = await axios.get(`http://127.0.0.1:8000/average-price-by-year/${selectedSuburb}`);
                    const { year_built, average_price } = response.data;

                    const formattedData = {
                        labels: year_built.map(year => year.toString()),
                        datasets: [
                            {
                                label: 'Average Price',
                                data: average_price,
                                borderColor: 'rgba(75,192,192,1)',
                                backgroundColor: 'rgba(75,192,192,0.2)',
                                fill: true,
                                tension: 0.1,
                            },
                        ],
                    };

                    setChartData(formattedData);
                } catch (error) {
                    console.error("Error fetching average price by year:", error);
                    setError("Failed to load average price by year data.");
                }
            };
            fetchAveragePriceByYear();
        }
    }, [selectedSuburb]);

    return (
        <div style={{ width: '600px', height: '400px', margin: 'auto' }}>
            <h3>Average Price by Year Built</h3>

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
                            x: { title: { display: true, text: 'Year Built' } },
                            y: {
                                beginAtZero: false,
                                title: { display: true, text: 'Average Price' },
                            },
                        },
                    }}
                />
            )}
        </div>
    );
};

export default AveragePriceByYearChart;

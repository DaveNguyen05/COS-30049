import React, { useEffect, useState } from 'react';
import { Bar } from 'react-chartjs-2';
import axios from 'axios';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const PriceDistributionChart = () => {
    const [suburbs, setSuburbs] = useState([]);               // For storing suburb options
    const [selectedSuburb, setSelectedSuburb] = useState('');  // For storing selected suburb
    const [priceDistributionData, setPriceDistributionData] = useState(null);
    const [error, setError] = useState(null);

    // Function to format large numbers with "K" and "M"
    const formatPrice = (value) => {
        if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`;
        if (value >= 1000) return `${(value / 1000).toFixed(1)}K`;
        return value.toString();
    };

    // Fetch available suburbs on initial render
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

    // Fetch price distribution data when a suburb is selected
    useEffect(() => {
        if (selectedSuburb) {
            const fetchPriceDistribution = async () => {
                try {
                    const response = await axios.get(`http://127.0.0.1:8000/price-distribution/${selectedSuburb}`);
                    const data = response.data;

                    const formattedData = {
                        labels: data.bin_edges.slice(0, -1).map((edge, i) =>
                            `${formatPrice(edge)} - ${formatPrice(data.bin_edges[i + 1])}`
                        ),
                        datasets: [
                            {
                                label: 'Price Distribution',
                                data: data.price_counts,
                                backgroundColor: 'rgba(75,192,192,0.6)',
                                borderColor: 'rgba(75,192,192,1)',
                                borderWidth: 1,
                            },
                        ],
                    };
                    setPriceDistributionData(formattedData);
                } catch (error) {
                    console.error("Error fetching price distribution:", error);
                    setError("Failed to load price distribution data.");
                }
            };
            fetchPriceDistribution();
        }
    }, [selectedSuburb]);

    return (
        <div style={{ width: '900px', height: '600px', margin: 'auto' }}>
            <h3>Price Distribution by Suburb</h3>

            {/* Dropdown for selecting a suburb */}
            <label>Select Suburb:</label>
            <select
                onChange={(e) => setSelectedSuburb(e.target.value)}
                value={selectedSuburb || ''}
            >
                <option value="">Select a Suburb</option>
                {suburbs.map((suburb) => (
                    <option key={suburb} value={suburb}>
                        {suburb}
                    </option>
                ))}
            </select>

            {error && <p style={{ color: 'red' }}>{error}</p>}
            {priceDistributionData ? (
                <Bar
                    data={priceDistributionData}
                    options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: {
                                title: {
                                    display: true,
                                    text: 'Price Range',
                                },
                                ticks: {
                                    maxRotation: 45,
                                    minRotation: 45,
                                },
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: 'Count',
                                },
                                beginAtZero: true,
                            },
                        },
                    }}
                />
            ) : (
                <p>Please select a suburb to view its price distribution.</p>
            )}
        </div>
    );
};

export default PriceDistributionChart;

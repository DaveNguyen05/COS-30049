import React, { useEffect, useState } from 'react';
import { Bar } from 'react-chartjs-2';
import axios from 'axios';

const PriceDistributionBySuburb = () => {
  const [chartData, setChartData] = useState({});

  useEffect(() => {
    axios.get('/api/property-prices-by-suburb')  // API endpoint to fetch the data
      .then(response => {
        const suburbs = response.data.map(item => item.suburb);
        const avgPrices = response.data.map(item => item.avg_price);

        setChartData({
          labels: suburbs,
          datasets: [
            {
              label: 'Average Price',
              data: avgPrices,
              backgroundColor: 'rgba(75, 192, 192, 0.6)',
            },
          ],
        });
      })
      .catch(error => console.error('Error fetching data:', error));
  }, []);

  return (
    <div>
      <Bar
        data={chartData}
        options={{
          responsive: true,
          scales: {
            y: { beginAtZero: true, title: { display: true, text: 'Average Price ($)' } },
            x: { title: { display: true, text: 'Suburb' } }
          }
        }}
      />
    </div>
  );
};

export default PriceDistributionBySuburb;

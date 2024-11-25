import React, { useState, useEffect } from 'react';
import axios from 'axios';


const PredictionForm = () => {
    const [suburbs, setSuburbs] = useState([]);
    const [formData, setFormData] = useState({});
    const [options, setOptions] = useState({});
    const [selectedSuburb, setSelectedSuburb] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [error, setError] = useState(null);
    const [selectedModel, setSelectedModel] = useState("forest"); // Default to Random Forest

    // Fetch unique suburbs on initial render
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

    // Fetch dynamic options based on selected suburb
    useEffect(() => {
        if (selectedSuburb) {
            const fetchOptions = async () => {
                try {
                    const response = await axios.get(`http://127.0.0.1:8000/options/${selectedSuburb}`);
                    setOptions(response.data);
                } catch (error) {
                    console.error("Error fetching options:", error);
                    setError("Failed to load options. Please check the server.");
                }
            };
            fetchOptions();
        }
    }, [selectedSuburb]);

    // Handle change for form inputs
    const handleChange = (e) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value,
        });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        try {
            const response = await axios.post('http://127.0.0.1:8000/predict', {
                Suburb: selectedSuburb,
                Rooms: parseInt(formData.rooms),
                Type: formData.type,
                Bathroom: parseInt(formData.bathrooms),
                Car: parseInt(formData.car),
                BuildingArea: formData.building_area ? parseFloat(formData.building_area) : null,
                Landsize: formData.landsize ? parseFloat(formData.landsize) : null,
                Regionname: formData.regionname,
                model_type: selectedModel
            });
            setPrediction(response.data.prediction);
            setError(null);
        } catch (error) {
            console.error('Error making prediction:', error);
            setError("Failed to get prediction. Please check the server.");
        }
    };

    return (
        <div>
            {error && <p style={{ color: 'red' }}>{error}</p>}
            <form onSubmit={handleSubmit}>
                <label>Select Suburb:</label>
                <select onChange={(e) => setSelectedSuburb(e.target.value)} value={selectedSuburb || ''}>
                    <option value="">Select a Suburb</option>
                    {suburbs.map((suburb) => (
                        <option key={suburb} value={suburb}>
                            {suburb}
                        </option>
                    ))}
                </select>

                {/* Show additional options once a suburb is selected */}
                {selectedSuburb && (
                    <>
                        <label>Rooms:</label>
                        <select name="rooms" onChange={handleChange} value={formData.rooms || ''}>
                            <option value="">Select Rooms</option>
                            {options.rooms?.map((room) => (
                                <option key={room} value={room}>
                                    {room}
                                </option>
                            ))}
                        </select>

                        <label>Type:</label>
                        <select name="type" onChange={handleChange} value={formData.type || ''}>
                            <option value="">Select Type</option>
                            {options.types?.map((type) => (
                                <option key={type} value={type}>
                                    {type}
                                </option>
                            ))}
                        </select>

                        <label>Bathrooms:</label>
                        <select name="bathrooms" onChange={handleChange} value={formData.bathrooms || ''}>
                            <option value="">Select Bathrooms</option>
                            {options.bathrooms?.map((bath) => (
                                <option key={bath} value={bath}>
                                    {bath}
                                </option>
                            ))}
                        </select>

                        <label>Car Spaces:</label>
                        <select name="car" onChange={handleChange} value={formData.car || ''}>
                            <option value="">Select Car Spaces</option>
                            {options.cars?.map((car) => (
                                <option key={car} value={car}>
                                    {car}
                                </option>
                            ))}
                        </select>

                        <label>Building Area:</label>
                        <input
                            name="building_area"
                            type="number"
                            placeholder="Building Area"
                            onChange={handleChange}
                            value={formData.building_area || ''}
                        />

                        <label>Landsize:</label>
                        <input
                            name="landsize"
                            type="number"
                            placeholder="Land Size"
                            onChange={handleChange}
                            value={formData.landsize || ''}
                        />

                        <label>Region:</label>
                        <select name="regionname" onChange={handleChange} value={formData.regionname || ''}>
                            <option value="">Select Region</option>
                            {options.regionname?.map((region) => (
                                <option key={region} value={region}>
                                    {region}
                                </option>
                            ))}
                        </select>

                        <label>Select Model:</label>
                        <select onChange={(e) => setSelectedModel(e.target.value)} value={selectedModel}>
                            <option value="forest">Random Forest</option>
                            <option value="gb">Gradient Boosting</option>
                        </select>

                        <button type="submit">Predict</button>
                    </>
                )}
            </form>
            {prediction && (
                <p>Predicted Price: {prediction}</p>

            )}
        </div>
    );
};

export default PredictionForm;

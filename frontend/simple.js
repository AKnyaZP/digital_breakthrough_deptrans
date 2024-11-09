// Install dependencies
// npm install nextui org --save
// npm install react-leaflet leaflet --save

import React from 'react';
import { NextUIProvider, Button, Grid, Card, Text, Spacer } from '@nextui-org/react';
import { MapContainer, TileLayer } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

const PedestrianPath = () => {
    const [selectedYear, setSelectedYear] = React.useState(null);

    const handleYearSelection = (year) => {
        setSelectedYear(year);
    };

    return (
        <NextUIProvider>
            <Grid.Container gap={2} justify="center">
                {/* Drag & Drop Area */}
                <Grid xs={12} sm={6} justify="center">
                    <Card hoverable clickable>
                        <Card.Body>
                            <input
                                type="file"
                                accept="image/*"
                                onChange={(e) => console.log(e.target.files)}
                                style={{ display: 'none' }}
                                id="fileInput"
                            />
                            <label htmlFor="fileInput">
                                <Text>Drag & Drop or Click to Upload</Text>
                            </label>
                        </Card.Body>
                    </Card>
                </Grid>

                {/* Year Selection Area */}
                <Grid xs={12} sm={6} justify="center">
                    <Card hoverable>
                        <Card.Body>
                            <Button.Group size="md">
                                <Button onClick={() => handleYearSelection(2018)}>2018</Button>
                                <Button onClick={() => handleYearSelection(2024)}>2024</Button>
                            </Button.Group>
                        </Card.Body>
                    </Card>
                </Grid>

                {/* Map Display Area */}
                {selectedYear && (
                    <Grid xs={12} justify="center">
                        <Card>
                            <Card.Body>
                                <MapContainer center={[38.9072, -77.0369]} zoom={13} style={{ height: '400px', width: '100%' }}>
                                    <TileLayer
                                        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                                        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                                    />
                                </MapContainer>
                                <Spacer y={0.5} />
                                <Text>Description of changes for {selectedYear}</Text>
                            </Card.Body>
                        </Card>
                    </Grid>
                )}
            </Grid.Container>
        </NextUIProvider>
    );
};

export default PedestrianPath;

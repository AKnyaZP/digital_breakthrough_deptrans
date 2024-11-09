import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Polyline } from 'react-leaflet';
import proj4 from 'proj4';

const MapWithRoutes = ({ coordinates }) => {
    const [convertedCoordinates, setConvertedCoordinates] = useState([]);

    useEffect(() => {
        // Преобразуем координаты из вашей системы в широту и долготу
        const converted = coordinates.map(coord => {
            // Замените "EPSG:3857" на систему координат вашей исходной системы
            const [lat, lon] = proj4('EPSG:3857', 'EPSG:4326', [coord.x, coord.y]);
            return [lon, lat];
        });
        setConvertedCoordinates(converted);
    }, [coordinates]);

    return (
        <MapContainer center={[48.8566, 2.3522]} zoom={13} style={{ height: '500px', width: '100%' }}>
            <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            {/* Отрисовка точек */}
            {convertedCoordinates.map((coord, index) => (
                <Marker key={index} position={coord} />
            ))}
            {/* Отрисовка маршрута, если есть хотя бы две точки */}
            {convertedCoordinates.length > 1 && (
                <Polyline positions={convertedCoordinates} color="blue" />
            )}
        </MapContainer>
    );
};

export default MapWithRoutes;

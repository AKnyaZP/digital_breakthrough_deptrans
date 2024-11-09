import React from 'react';
import { MapContainer, TileLayer } from 'react-leaflet';

const MapComparison = ({ mapData, onSelectMap }) => {
  return (
    <div className="map-container" style={{ display: 'flex', justifyContent: 'space-between', marginTop: '20px' }}>
      {mapData.map((map, index) => (
        <div key={index} onClick={() => onSelectMap(map)} className="map-item" style={{ cursor: 'pointer' }}>
          <h3>{map.year}</h3>
          <MapContainer
            center={[38.89511, -77.03637]} // Установите координаты центра карты
            zoom={12} // Установите уровень масштабирования для лучшего отображения
            style={{ height: '200px', width: '300px' }}
          >
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
          </MapContainer>
        </div>
      ))}
    </div>
  );
};

export default MapComparison;

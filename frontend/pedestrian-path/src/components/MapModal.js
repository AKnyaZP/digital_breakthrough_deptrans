import React from 'react';
import { MapContainer, TileLayer } from 'react-leaflet';

function MapModal({ map, onClose }) {
  return (
    <div className="map-modal">
      <button onClick={onClose}>Закрыть</button>
      <h3>{map.year} - Подробные изменения</h3>
      <MapContainer center={[38.89511, -77.03637]} zoom={13} style={{ height: "400px", width: "100%" }}>
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      </MapContainer>
      <p>{map.description}</p>
    </div>
  );
}

export default MapModal;

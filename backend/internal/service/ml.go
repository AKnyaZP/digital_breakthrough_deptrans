package service

import (
	"context"
	"deptrans-backend/internal/repository"
)

type MLService struct {
	geoDataRepository repository.GeoData
}

func NewMLService(geoDataRepository repository.GeoData) *MLService {
	return &MLService{geoDataRepository: geoDataRepository}
}

func (s *MLService) Predict(ctx context.Context) {
}

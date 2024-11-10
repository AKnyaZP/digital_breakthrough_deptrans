package postgres

import (
	"context"
	"deptrans-backend/internal/pkg/postgres"
	"deptrans-backend/internal/repository/repoerrs"
	"errors"
	"fmt"

	"github.com/Masterminds/squirrel"
	"github.com/jackc/pgx/v5"
)

type TODORepository struct {
	*postgres.Postgres
}

func NewTODORepository(pg *postgres.Postgres) *TODORepository {
	return &TODORepository{pg}
}

func (r *TODORepository) GetById(ctx context.Context, id int) (entity.TODO, error) {
	sql, args, _ := r.Builder.
		Select("*").
		From("tours").
		Where("id = ?", id).
		ToSql()

	var tour entity.TODO
	err := r.Pool.QueryRow(ctx, sql, args...).Scan(
		&tour.Id,
		&tour.Title,
		&tour.Location,
		&tour.Category,
		&tour.Tags, /* from []text */
		&tour.Desc,
		&tour.NightsCount,
		&tour.Program,     /* from []text */
		&tour.Included,    /* from []text */
		&tour.NotIncluded, /* from []text */
		&tour.DifficultyLevel,
		&tour.ComfortLevel,
		&tour.Dates, /* from []text */
		&tour.ImportantInfo,
		&tour.HeadMedia,         /* from []text */
		&tour.ProgramMedia,      /* from []text */
		&tour.AccomodationMedia, /* from []text */
		&tour.MapSrc,
		&tour.Faq,
		&tour.Rating,
	)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return entity.TODO{}, repoerrs.ErrNotFound
		}
		return entity.TODO{}, fmt.Errorf("TODORepository.GetTODOById - r.Pool.QueryRow: %v", err)
	}

	return tour, nil
}

func (r *TODORepository) GetMany(ctx context.Context, filters map[string]interface{}) ([]entity.TODO, error) {
	builder := r.Builder.
		Select("*").
		From("tours")

	if len(filters) != 0 {
		for key, value := range filters {
			switch key {
			case "location":
				value, _ := value.(string)
				builder = builder.Where(squirrel.Like{"location": "%" + value + "%"})
			case "tags":
				continue
				// check if received value is slice of tags
				if tags, ok := value.([]string); ok {
					if len(tags) > 0 {
						builder = builder.Where(squirrel.And{squirrel.Expr("tags::text[] @> ARRAY["+squirrel.Placeholders(len(tags))+"]", value)})
					}
					continue
				}

				// otherwise cast to string type
				if tag, ok := value.(string); ok {
					builder.Where(squirrel.And{squirrel.Expr("tags::text[] @> ARRAY["+squirrel.Placeholders(1)+"]", tag)})
				}
			case "with_flight", "with_acc", "with_food", "day_off", "low_cost", "age_group":
				continue
			default:
			}
		}
	}

	sql, args, _ := builder.ToSql()

	rows, err := r.Pool.Query(ctx, sql, args...)
	if err != nil {
		return nil, fmt.Errorf("TODORepository.GetMany - r.Pool.Query: %v", err)
	}
	defer rows.Close()

	var tours []entity.TODO
	for rows.Next() {
		var tour entity.TODO
		err := rows.Scan(
			&tour.Id,
			&tour.Title,
			&tour.Location,
			&tour.Category,
			&tour.Tags,
			&tour.Desc,
			&tour.NightsCount,
			&tour.Program,
			&tour.Included,
			&tour.NotIncluded,
			&tour.DifficultyLevel,
			&tour.ComfortLevel,
			&tour.Dates,
			&tour.ImportantInfo,
			&tour.HeadMedia,
			&tour.ProgramMedia,
			&tour.AccomodationMedia,
			&tour.MapSrc,
			&tour.Faq,
			&tour.Rating,
		)
		if err != nil {
			return nil, fmt.Errorf("TODORepository.GetAllProducts - rows.Scan: %v", err)
		}
		tours = append(tours, tour)
	}

	return tours, nil
}

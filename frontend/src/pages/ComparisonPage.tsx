import { Box, Card, CardContent, Typography, Button, List, ListItem, ListItemText, ListItemSecondaryAction, IconButton, Checkbox } from '@mui/material';
import { Delete as DeleteIcon } from '@mui/icons-material';
import { useAppDispatch, useAppSelector, comparisonActions } from '../store';

export default function ComparisonPage() {
  const dispatch = useAppDispatch();
  const { savedRuns, selectedRunIds } = useAppSelector((state) => state.comparison);

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Compare Simulation Runs
      </Typography>

      <Card>
        <CardContent>
          {savedRuns.length === 0 ? (
            <Typography color="text.secondary">
              No saved runs yet. Complete a simulation and save it to compare results.
            </Typography>
          ) : (
            <List>
              {savedRuns.map((run) => (
                <ListItem key={run.id}>
                  <Checkbox
                    checked={selectedRunIds.includes(run.id)}
                    onChange={() => dispatch(comparisonActions.toggleRunSelection(run.id))}
                  />
                  <ListItemText
                    primary={run.name}
                    secondary={new Date(run.timestamp).toLocaleString()}
                  />
                  <ListItemSecondaryAction>
                    <IconButton
                      edge="end"
                      onClick={() => dispatch(comparisonActions.removeSavedRun(run.id))}
                    >
                      <DeleteIcon />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
          )}

          {selectedRunIds.length > 0 && (
            <Box sx={{ mt: 2 }}>
              <Button
                variant="outlined"
                onClick={() => dispatch(comparisonActions.clearSelection())}
              >
                Clear Selection
              </Button>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* TODO: Add comparison chart when runs are selected */}
    </Box>
  );
}

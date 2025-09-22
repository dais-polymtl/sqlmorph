# Load and export environment variables from the .env file
while read -r line || [[ -n $line ]]; do
  # Skip empty lines and lines starting with #
  if [[ ! $line || $line == \#* ]]; then
    continue
  fi
  export "$line"
done < .env

echo "Environment variables loaded."

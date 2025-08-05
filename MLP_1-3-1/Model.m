% Einfaches MLP mit zwei Schichten in MATLAB mit Backpropagation

% Zufällige Daten generieren
inputSize = 1; % Anzahl der Eingabeneuronen
hiddenSize = 5; % Anzahl der Neuronen in der versteckten Schicht
outputSize = 1; % Anzahl der Ausgabeneuronen

% Zufällige Eingabedaten (10 Beispiele)
X = rand(10, inputSize);
% Zufällige Zielwerte (10 Beispiele)
Y = rand(10, outputSize);

% Zufällige Gewichte und Biases initialisieren
W1 = rand(inputSize, hiddenSize); % Gewichte von Eingabe zu versteckter Schicht
b1 = rand(1, hiddenSize); % Biases für versteckte Schicht
W2 = rand(hiddenSize, outputSize); % Gewichte von versteckter zu Ausgabeschicht
b2 = rand(1, outputSize); % Biases für Ausgabeschicht

% Lernrate
learningRate = 0.01;

% Anzahl der Epochen
epochs = 1000;

% Training des MLP
for epoch = 1:epochs
    % Vorwärtsdurchlauf
    % Versteckte Schicht
    Z1 = X * W1 + b1; % Lineare Kombination
    A1 = tanh(Z1); % Aktivierungsfunktion (Tanh)

    % Ausgabeschicht
    Z2 = A1 * W2 + b2; % Lineare Kombination (keine Aktivierungsfunktion)

    % Verlustfunktion (Mean Squared Error)
    loss = mean(sum((Z2 - Y).^2, 2));

    % Rückpropagation
    % Fehler im Ausgabelayer
    dZ2 = Z2 - Y;
    dW2 = (A1' * dZ2) / size(X, 1);
    db2 = sum(dZ2, 1) / size(X, 1);

    % Fehler im versteckten Layer
    dA1 = dZ2 * W2';
    dZ1 = dA1 .* tanh_derivative(Z1);
    dW1 = (X' * dZ1) / size(X, 1);
    db1 = sum(dZ1, 1) / size(X, 1);

    % Parameteraktualisierung
    W1 = W1 - learningRate * dW1;
    b1 = b1 - learningRate * db1;
    W2 = W2 - learningRate * dW2;
    b2 = b2 - learningRate * db2;

    % Ausgabe des Verlusts alle 100 Epochen
    if mod(epoch, 100) == 0
        fprintf('Epoch %d, Loss: %.4f\n', epoch, loss);
    end
end

% Ausgabe des trainierten MLP
disp('Trainiertes MLP:');
disp(Z2);

% Tanh-Aktivierungsfunktion
function y = tanh(x)
    y = (exp(x) - exp(-x)) ./ (exp(x) + exp(-x));
end

% Ableitung der Tanh-Aktivierungsfunktion
function y = tanh_derivative(x)
    y = 1 - tanh(x).^2;
end
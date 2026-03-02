import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../styles" as Styles

Dialog {
    id: dlg
    title: ""
    modal: true
    standardButtons: Dialog.NoButton
    width: 420
    height: 380
    dim: true

    property string modelId: ""
    property string device: ""
    property int threads: 4
    property string dtype: "auto"
    property int configPort: 0

    signal saved(string mid, string device, int threads, string dtype, int port)

    Overlay.modal: Rectangle { color: "#80000000" }

    background: Rectangle {
        color: Styles.Theme.bgPopup
        border.color: Styles.Theme.border
        radius: Styles.Theme.borderRadiusLarge
    }

    contentItem: ColumnLayout {
        spacing: 0

        // Title bar
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 40
            color: "transparent"

            Text {
                anchors.centerIn: parent
                text: qsTr("Model Launch Settings")
                font.pixelSize: Styles.Theme.fontSizeLarge
                font.weight: Font.DemiBold
                color: Styles.Theme.textBright
            }
            Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: Styles.Theme.border }
        }

        // Form
        GridLayout {
            columns: 2
            columnSpacing: Styles.Theme.paddingMedium
            rowSpacing: Styles.Theme.paddingMedium
            Layout.fillWidth: true
            Layout.margins: Styles.Theme.paddingLarge

            Text { text: qsTr("Device"); font.pixelSize: Styles.Theme.fontSizeMedium; color: Styles.Theme.textSecondary }
            ComboBox {
                id: deviceCombo
                model: ["", "cpu", "cuda", "numa"]
                currentIndex: Math.max(0, model.indexOf(dlg.device))
                Layout.fillWidth: true
                palette.window: Styles.Theme.bgInput
                palette.text: Styles.Theme.textPrimary
                palette.buttonText: Styles.Theme.textPrimary
            }

            Text { text: qsTr("Threads"); font.pixelSize: Styles.Theme.fontSizeMedium; color: Styles.Theme.textSecondary }
            SpinBox {
                id: threadsSpin; from: 1; to: 128; value: dlg.threads; Layout.fillWidth: true
                palette.window: Styles.Theme.bgInput; palette.text: Styles.Theme.textPrimary
                palette.buttonText: Styles.Theme.textPrimary
            }

            Text { text: qsTr("Data Type"); font.pixelSize: Styles.Theme.fontSizeMedium; color: Styles.Theme.textSecondary }
            ComboBox {
                id: dtypeCombo
                model: ["auto", "float16", "float32", "int8", "int4"]
                currentIndex: Math.max(0, model.indexOf(dlg.dtype))
                Layout.fillWidth: true
                palette.window: Styles.Theme.bgInput; palette.text: Styles.Theme.textPrimary
                palette.buttonText: Styles.Theme.textPrimary
            }

            Text { text: qsTr("Port"); font.pixelSize: Styles.Theme.fontSizeMedium; color: Styles.Theme.textSecondary }
            SpinBox {
                id: portSpin; from: 0; to: 65535; value: dlg.configPort; Layout.fillWidth: true
                palette.window: Styles.Theme.bgInput; palette.text: Styles.Theme.textPrimary
                palette.buttonText: Styles.Theme.textPrimary
            }

            Item { Layout.columnSpan: 2 }
            Text {
                text: "Port 0 = auto"
                font.pixelSize: Styles.Theme.fontSizeTiny
                color: Styles.Theme.textDisabled
                Layout.columnSpan: 2
                Layout.alignment: Qt.AlignRight
            }
        }

        Item { Layout.fillHeight: true }

        // Footer
        Rectangle {
            Layout.fillWidth: true; Layout.preferredHeight: 1; color: Styles.Theme.border
        }
        RowLayout {
            Layout.fillWidth: true
            Layout.margins: Styles.Theme.paddingMedium
            spacing: Styles.Theme.spacingSmall

            Item { Layout.fillWidth: true }

            Rectangle {
                width: cancelLbl.implicitWidth + 24; height: 30; radius: Styles.Theme.borderRadius
                color: cancelMa.containsMouse ? Styles.Theme.bgInput : "transparent"
                border.color: Styles.Theme.border; border.width: 1
                Text { id: cancelLbl; anchors.centerIn: parent; text: qsTr("Cancel"); font.pixelSize: Styles.Theme.fontSizeMedium; color: Styles.Theme.textPrimary }
                MouseArea { id: cancelMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: dlg.reject() }
            }

            Rectangle {
                width: saveLbl.implicitWidth + 24; height: 30; radius: Styles.Theme.borderRadius
                color: saveMa.containsMouse ? Styles.Theme.primaryHover : Styles.Theme.primaryColor
                Text { id: saveLbl; anchors.centerIn: parent; text: qsTr("Save"); font.pixelSize: Styles.Theme.fontSizeMedium; color: Styles.Theme.textOnAccent }
                MouseArea {
                    id: saveMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                    onClicked: { dlg.saved(dlg.modelId, deviceCombo.currentText, threadsSpin.value, dtypeCombo.currentText, portSpin.value); dlg.accept() }
                }
            }
        }
    }
}
